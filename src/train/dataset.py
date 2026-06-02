import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import transformers
from torch.utils.data import Dataset
import json

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
                 max_len: int = 8192, language: str | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items: list[dict] = []

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

                    pairs = []
                    for ann in entry.get("attention_edges", []):
                        qi = ann.get("src", ann.get("source", ann.get("token_i_idx", -1)))
                        qj = ann.get("dst", ann.get("target", ann.get("token_j_idx", -1)))
                        qi = int(qi)
                        qj = int(qj)
                        if 0 <= qi < qj < len(input_ids):
                            pairs.append((qi, qj))

                    self.items.append({
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                        "annot_pairs": torch.tensor(pairs, dtype=torch.long)
                        if pairs else torch.zeros(0, 2, dtype=torch.long),
                    })
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

                # Build annotation pairs
                pairs = []
                for ann in annotated_edges:
                    qi = ann.get("token_i_idx", -1)
                    qj = ann.get("token_j_idx", -1)
                    if 0 <= qi < qj < len(input_ids):
                        pairs.append((qi, qj))

                self.items.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "annot_pairs": torch.tensor(pairs, dtype=torch.long)
                    if pairs else torch.zeros(0, 2, dtype=torch.long),
                })

        logger.info(f"Loaded {len(self.items)} samples from {data_path}")

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
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "annot_pairs": annot_pairs_list,  # list of tensors
        }
