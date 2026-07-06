import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import transformers
from torch.utils.data import Dataset
import json

# Make the src/ root importable so the data-side edge utilities resolve the same
# way as elsewhere in the repo (``from data.<mod> import ...``).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.edge_augment import augment_edges, node_target_weights  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Chat template constants (must match postprocessing.py) ────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."
CHAT_PREFIX   = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
CHAT_MIDDLE   = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_SUFFIX   = "<|im_end|>\n"

IGNORE_INDEX = -100

class AnnotatedSFTDataset(Dataset):
    """
    Each item:
        input_ids  : LongTensor [seq_len]
        labels     : LongTensor [seq_len]  (IGNORE_INDEX for input tokens)
        annot_pairs: LongTensor [N, 2]     (qi, qj) — may be empty
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int = 8192, language: str | None = None,
                 edge_augment: bool = False, edge_augment_decay: float = 0.5,
                 edge_augment_max_hops: int = 0, edge_augment_node_weight: bool = False,
                 edge_augment_mode: str = "directed",
                 token_select: bool = False, token_select_threshold: float = 2.0,
                 token_select_keep_special: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items: list[dict] = []

        # ── Edge-label augmentation (config-gated, default OFF) ────────────────
        # When enabled, edges are densified by transitive closure with a per-hop
        # decay weight (see data/edge_augment.py). Disabled => identical behavior.
        self.edge_augment = bool(edge_augment)
        self.edge_augment_decay = float(edge_augment_decay)
        self.edge_augment_max_hops = int(edge_augment_max_hops)
        self.edge_augment_node_weight = bool(edge_augment_node_weight)
        self.edge_augment_mode = str(edge_augment_mode or "directed")
        if self.edge_augment:
            logger.info(
                f"Edge augmentation ON: {self.edge_augment_mode} closure, decay={self.edge_augment_decay}, "
                f"max_hops={self.edge_augment_max_hops or 'inf'}, "
                f"node_weight={self.edge_augment_node_weight}"
            )

        # ── Teacher-gated informative-token selection (config-gated, default OFF)
        # When enabled, completion tokens whose precomputed teacher NLL
        # (``comp_teacher_nll`` field) exceeds the threshold are "missing-info"
        # (uninferable from the input) and are EXCLUDED from the loss by setting
        # their label to IGNORE_INDEX. Special tokens (EOS/im_end) are kept.
        self.token_select = bool(token_select)
        self.token_select_threshold = float(token_select_threshold)
        self.token_select_keep_special = bool(token_select_keep_special)
        self._special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self._ts_excluded = 0
        self._ts_total = 0
        if self.token_select:
            logger.info(
                f"Token selection ON: exclude completion tokens with teacher NLL > "
                f"{self.token_select_threshold} (keep_special={self.token_select_keep_special})"
            )

        prefix_len = len(CHAT_PREFIX)
        middle = CHAT_MIDDLE
        suffix = CHAT_SUFFIX

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                if "input_ids" in entry and ("label" in entry or "labels" in entry):
                    input_ids = [int(x) for x in entry["input_ids"]]
                    labels = [int(x) for x in entry.get("label", entry.get("labels", []))]
                    if not input_ids or len(input_ids) != len(labels):
                        continue
                    if len(input_ids) > max_len:
                        continue

                    if self.token_select:
                        self._apply_token_select(entry, input_ids, labels)

                    self.items.append(
                        self._make_item(input_ids, labels, entry.get("attention_edges", []))
                    )
                    continue

                task_id = entry["task_id"]
                lang = task_id.split("/")[0]
                if language is not None and lang.lower() != language.lower():
                    continue

                sft_input = entry.get("sft_input", "")
                tokens = entry.get("qwen_tokens", [])
                annotated_edges = entry.get("qwen_annotations", [])

                if not tokens:
                    continue

                # Build input_ids from qwen_tokens
                input_ids = [t["token_id"] for t in tokens]
                if len(input_ids) > max_len:
                    continue

                # output starts right after CHAT_PREFIX + sft_input + CHAT_MIDDLE
                output_char_start = len(CHAT_PREFIX) + len(sft_input) + len(CHAT_MIDDLE)

                # Find first qwen token whose char_start >= output_char_start
                output_token_start = len(input_ids)  # default: no output tokens
                for idx, tok in enumerate(tokens):
                    if tok["char_start"] >= output_char_start:
                        output_token_start = idx
                        break

                # Build labels
                labels = [IGNORE_INDEX] * output_token_start + input_ids[output_token_start:]

                self.items.append(self._make_item(input_ids, labels, annotated_edges))

        logger.info(f"Loaded {len(self.items)} samples from {data_path}")
        if self.token_select and self._ts_total:
            logger.info(
                f"Token selection: excluded {self._ts_excluded}/{self._ts_total} scored "
                f"completion tokens ({100*self._ts_excluded/self._ts_total:.1f}%) as missing-info "
                f"(teacher NLL > {self.token_select_threshold})"
            )

    def _apply_token_select(self, entry, input_ids, labels):
        """Exclude 'missing-info' completion tokens from the loss in-place.

        A completion token whose precomputed teacher NLL exceeds the threshold is
        not inferable from the input; its label is set to IGNORE_INDEX so the CE
        loss skips it. Special tokens (EOS/im_end) are kept when configured.
        """
        for pos, nll in entry.get("comp_teacher_nll", []) or []:
            pos = int(pos)
            if not (0 <= pos < len(labels)) or labels[pos] == IGNORE_INDEX:
                continue
            self._ts_total += 1
            if float(nll) <= self.token_select_threshold:
                continue
            if self.token_select_keep_special and int(input_ids[pos]) in self._special_ids:
                continue
            labels[pos] = IGNORE_INDEX
            self._ts_excluded += 1

    def _build_annot(self, raw_edges, n_tokens, labels):
        """Return ``(pairs, weights, node_weight)`` for one sample.

        ``pairs`` is the list of ``(qi, qj)`` edges (qi < qj). With augmentation
        OFF this reproduces the legacy edge list exactly and ``weights`` /
        ``node_weight`` are ``None`` (so the item dict is byte-for-byte legacy).
        With augmentation ON, edges are densified by transitive closure;
        ``weights`` is the per-edge decay weight and ``node_weight`` (only when
        ``edge_augment_node_weight``) is a length-``n_tokens`` vector giving each
        position's max weight into any target token (for weight-aware cfmask).
        """
        # Normalize raw edges to a uniform {src, dst, subtype} list (handles both
        # the compact `attention_edges` and the qwen `token_i/j_idx` schemas).
        norm = []
        for ann in raw_edges:
            qi = ann.get("src", ann.get("source", ann.get("token_i_idx", -1)))
            qj = ann.get("dst", ann.get("target", ann.get("token_j_idx", -1)))
            try:
                qi = int(qi)
                qj = int(qj)
            except (TypeError, ValueError):
                continue
            norm.append({"src": qi, "dst": qj, "subtype": ann.get("subtype", "edge")})

        if self.edge_augment:
            edges = augment_edges(
                norm,
                decay=self.edge_augment_decay,
                max_hops=self.edge_augment_max_hops,
                n_tokens=n_tokens,
                mode=self.edge_augment_mode,
            )
        else:
            edges = norm  # weight implicitly 1.0; kept identical to legacy path

        pairs, weights = [], []
        for e in edges:
            qi, qj = int(e["src"]), int(e["dst"])
            if 0 <= qi < qj < n_tokens:
                pairs.append((qi, qj))
                weights.append(float(e.get("weight", 1.0)))

        node_weight = None
        if self.edge_augment and self.edge_augment_node_weight:
            tgt = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
            nw = node_target_weights(edges, tgt)
            node_weight = [0.0] * n_tokens
            for p, w in nw.items():
                if 0 <= p < n_tokens:
                    node_weight[p] = float(w)

        if not self.edge_augment:
            weights = None  # legacy path surfaces no weights
        return pairs, weights, node_weight

    def _make_item(self, input_ids, labels, raw_edges):
        """Build one dataset item dict, attaching augmentation tensors if on."""
        n = len(input_ids)
        pairs, weights, node_weight = self._build_annot(raw_edges, n, labels)
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "annot_pairs": torch.tensor(pairs, dtype=torch.long)
            if pairs else torch.zeros(0, 2, dtype=torch.long),
        }
        if weights is not None:
            item["annot_weights"] = (torch.tensor(weights, dtype=torch.float)
                                     if weights else torch.zeros(0, dtype=torch.float))
        if node_weight is not None:
            item["node_weight"] = torch.tensor(node_weight, dtype=torch.float)
        return item

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── Collator ──────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorForAnnotatedSFT:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = [inst["input_ids"] for inst in instances]
        labels_list = [inst["labels"] for inst in instances]
        annot_pairs_list = [inst.get("annot_pairs", torch.zeros(0, 2, dtype=torch.long))
                            for inst in instances]
        # Pad input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True,
            padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # annot_pairs: list of [N_i, 2] tensors, one per sample in batch
        # We keep them as a list (ragged) — the trainer will handle per-sample.
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "annot_pairs": annot_pairs_list,  # list of tensors
        }

        # Optional augmentation fields (present only when edge_augment is on).
        # Kept ragged/padded and added conditionally so the default path is
        # byte-for-byte identical to before.
        if "annot_weights" in instances[0]:
            batch["annot_weights"] = [
                inst.get("annot_weights", torch.zeros(0, dtype=torch.float))
                for inst in instances
            ]
        if "node_weight" in instances[0]:
            node_weight = torch.nn.utils.rnn.pad_sequence(
                [inst["node_weight"] for inst in instances],
                batch_first=True, padding_value=0.0,
            )
            batch["node_weight"] = node_weight
        return batch
