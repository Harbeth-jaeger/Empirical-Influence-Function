from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import torch
import transformers
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from attn_viz import AttentionVisualizationCallback
from dataset import AnnotatedSFTDataset, DataCollatorForAnnotatedSFT
from loss import canonical_saliency_loss_type, saliency_loss_display_name, saliency_loss_from_outputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Compatibility shim for older torch builds used with newer Transformers FSDP code.
try:
    import torch.distributed.fsdp as _fsdp
    if not hasattr(_fsdp, "register_fsdp_forward_method"):
        def _register_fsdp_forward_method(model, method_name):
            return None
        _fsdp.register_fsdp_forward_method = _register_fsdp_forward_method
except Exception:
    pass



# ── Custom Trainer ────────────────────────────────────────────────────────────


class AnnotatedSFTTrainer(Trainer):
    """
        NTP loss + saliency loss.
    """
    def __init__(self, *args,
                 saliency_lambda: float = 0.1,
                 saliency_alpha: float = 1.5,
                 saliency_eps: float = 1e-8,
                 saliency_floor_eps: float = 0.0,
                 saliency_floor_eps_mode: str = "ema_quantile",
                 saliency_floor_quantile: float = 0.75,
                 saliency_floor_ema_beta: float = 0.95,
                 saliency_floor_min_eps: float = 1e-8,
                 saliency_floor_warmup_steps: int = 10,
                 saliency_floor_logit_eps: float | None = None,
                 saliency_loss_type: str = "softmax_margin",
                 loss_mode: str = "ce_saliency",
                 saliency_detail_log_path: str = "",
                 saliency_detail_log_steps: int = 0,
                 saliency_detail_top_k: int = 10,
                 saliency_margin_plus: float = 2.08,
                 saliency_margin_minus: float = 0.41,
                 saliency_margin_gamma: float = 2.0,
                 saliency_neg_weight: float = 0.5,
                 saliency_neg_hard_only: bool = False,
                 saliency_neg_sample_k: int = 0,
                 saliency_source_chunk_size: int = 16,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.saliency_lambda = saliency_lambda
        self.saliency_alpha = saliency_alpha
        self.saliency_eps = saliency_eps
        self.saliency_floor_eps = saliency_floor_eps
        self.saliency_floor_eps_mode = (saliency_floor_eps_mode or "fixed").strip().lower()
        self.saliency_floor_quantile = saliency_floor_quantile
        self.saliency_floor_ema_beta = saliency_floor_ema_beta
        self.saliency_floor_min_eps = saliency_floor_min_eps
        self.saliency_floor_warmup_steps = saliency_floor_warmup_steps
        self.saliency_floor_logit_eps = saliency_floor_logit_eps
        self.saliency_loss_type = canonical_saliency_loss_type(saliency_loss_type)
        self.saliency_floor_eps_ema: float | None = None
        self.saliency_floor_step = 0
        self.loss_mode = loss_mode
        self.saliency_detail_log_path = saliency_detail_log_path
        self.saliency_detail_log_steps = saliency_detail_log_steps
        self.saliency_detail_top_k = saliency_detail_top_k
        self.saliency_margin_plus = saliency_margin_plus
        self.saliency_margin_minus = saliency_margin_minus
        self.saliency_margin_gamma = saliency_margin_gamma
        self.saliency_neg_weight = saliency_neg_weight
        self.saliency_neg_hard_only = saliency_neg_hard_only
        self.saliency_neg_sample_k = int(saliency_neg_sample_k or 0)
        self.saliency_source_chunk_size = max(1, int(saliency_source_chunk_size or 16))
        if self.saliency_detail_log_path and self.is_world_process_zero():
            os.makedirs(os.path.dirname(self.saliency_detail_log_path), exist_ok=True)

        # Step 1 needs real attention probabilities; SDPA / FA-2 hide them.
        attn_impl = getattr(self.model.config, "_attn_implementation", None)
        assert attn_impl == "eager", (
            f"AnnotatedSFTTrainer needs attn_implementation='eager', got {attn_impl!r}. "
            "Re-init the model with attn_implementation='eager'."
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        annot_pairs_batch: list[torch.Tensor] = inputs.pop("annot_pairs")

        needs_saliency = self.loss_mode != "ce_only"
        forward_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "output_attentions": needs_saliency,
            "output_hidden_states": needs_saliency,
        }
        if self.loss_mode != "saliency_only":
            forward_kwargs["labels"] = inputs["labels"]
        outputs = model(**forward_kwargs)
        loss_device = outputs.logits.device
        loss_dtype = outputs.logits.dtype

        if self.loss_mode == "saliency_only":
            ntp_loss = torch.zeros((), device=loss_device, dtype=loss_dtype)
        else:
            ntp_loss = outputs.loss
            if ntp_loss.dim() > 0:
                ntp_loss = ntp_loss.mean()

        try:
            outputs.logits = None
        except Exception:
            pass

        diag = None
        saliency_loss = torch.zeros((), device=loss_device, dtype=loss_dtype)
        if self.loss_mode != "ce_only":
            floor_step = self.saliency_floor_step + 1
            diag = saliency_loss_from_outputs(
                model,
                outputs,
                annot_pairs_batch,
                alpha=self.saliency_alpha,
                eps=self.saliency_eps,
                floor_eps=self.saliency_floor_eps,
                floor_eps_mode=self.saliency_floor_eps_mode,
                floor_eps_quantile=self.saliency_floor_quantile,
                floor_eps_ema_beta=self.saliency_floor_ema_beta,
                prev_floor_eps=self.saliency_floor_eps_ema,
                floor_eps_min=self.saliency_floor_min_eps,
                floor_eps_step=floor_step,
                floor_eps_warmup_steps=self.saliency_floor_warmup_steps,
                floor_logit_eps=self.saliency_floor_logit_eps,
                loss_type=self.saliency_loss_type,
                margin_plus=self.saliency_margin_plus,
                margin_minus=self.saliency_margin_minus,
                margin_gamma=self.saliency_margin_gamma,
                neg_weight=self.saliency_neg_weight,
                neg_hard_only=self.saliency_neg_hard_only,
                neg_sample_k=self.saliency_neg_sample_k,
                source_chunk_size=self.saliency_source_chunk_size,
            )
            saliency_loss = diag.loss
            self.saliency_floor_step = floor_step
            if (
                self.saliency_loss_type != "softmax"
                and self.saliency_floor_logit_eps is None
                and self.saliency_floor_eps_mode == "ema_quantile"
                and diag.n_queries > 0
            ):
                self.saliency_floor_eps_ema = diag.floor_eps

            if diag.n_samples > 0:
                loss_label = saliency_loss_display_name(diag.loss_type)
                logger.info(
                    f"[saliency:{loss_label}] C̄={diag.avg_C:.4f}  N̄={diag.avg_N:.4f}  "
                    f"ratio={diag.avg_ratio:.3f}  tau={self.saliency_alpha}  "
                    f"eps_num={self.saliency_eps:.4g}  floor_mode={diag.floor_eps_mode}  "
                    f"floor_step={diag.floor_eps_step}  warmup={diag.floor_eps_warmup_steps}  "
                    f"eps_floor_effective={diag.floor_eps:.4g}  eps_floor_batch={diag.batch_floor_eps:.4g}  "
                    f"eps_floor_logit={diag.floor_logit_eps:.4g}  floor_kind={diag.floor_eps_kind}  "
                    f"loss={diag.loss.item():.4f}  #queries={diag.n_queries}"
                )

        if self.loss_mode == "ce_saliency":
            total_loss = ntp_loss + self.saliency_lambda * saliency_loss
        elif self.loss_mode == "saliency_only":
            total_loss = saliency_loss
        elif self.loss_mode == "ce_only":
            total_loss = ntp_loss
        else:
            raise ValueError(f"Unsupported loss_mode={self.loss_mode!r}")

        detail = None
        should_detail_log = (
            self.loss_mode != "ce_only"
            and self.saliency_detail_log_path
            and self.saliency_detail_log_steps > 0
            and self.state.global_step % self.saliency_detail_log_steps == 0
            and self.is_world_process_zero()
        )
        if should_detail_log:
            from saliency_diagnostics import saliency_details_from_outputs

            detail = saliency_details_from_outputs(
                model,
                outputs,
                annot_pairs_batch,
                alpha=self.saliency_alpha,
                eps=self.saliency_eps,
                floor_eps=diag.floor_eps if diag is not None else self.saliency_floor_eps,
                floor_logit_eps=diag.floor_logit_eps if diag is not None and diag.floor_eps_kind == "logit" else None,
                top_k=self.saliency_detail_top_k,
            )
            with open(self.saliency_detail_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": int(self.state.global_step),
                    "loss_mode": self.loss_mode,
                    "saliency_loss_type": self.saliency_loss_type,
                    "ntp_loss": float(ntp_loss.detach().cpu()),
                    "saliency_loss": float(saliency_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "teacher_diag": {
                        "loss_type": diag.loss_type if diag is not None else self.saliency_loss_type,
                        "Cbar": diag.avg_C if diag is not None else 0.0,
                        "Nbar": diag.avg_N if diag is not None else 0.0,
                        "ratio": diag.avg_ratio if diag is not None else 0.0,
                        "eps_num": self.saliency_eps,
                        "floor_eps": diag.floor_eps if diag is not None else self.saliency_floor_eps,
                        "batch_floor_eps": diag.batch_floor_eps if diag is not None else 0.0,
                        "floor_logit_eps": diag.floor_logit_eps if diag is not None else 0.0,
                        "floor_eps_kind": diag.floor_eps_kind if diag is not None else "saliency",
                        "floor_eps_mode": diag.floor_eps_mode if diag is not None else self.saliency_floor_eps_mode,
                        "floor_eps_step": diag.floor_eps_step if diag is not None else self.saliency_floor_step,
                        "floor_eps_warmup_steps": diag.floor_eps_warmup_steps if diag is not None else self.saliency_floor_warmup_steps,
                        "n_queries": diag.n_queries if diag is not None else 0,
                        "n_samples": diag.n_samples if diag is not None else 0,
                    },
                    "strict_causal_detail": {
                        "Cbar": detail.avg_C,
                        "Nbar": detail.avg_N,
                        "ratio": detail.avg_ratio,
                        "loss": detail.loss,
                        "active_margin_rate": detail.active_margin_rate,
                        "hit_at_k": detail.hit_at_k,
                        "recall_at_k": detail.recall_at_k,
                        "precision_at_k": detail.precision_at_k,
                        "map_at_k": detail.map_at_k,
                        "top_k": self.saliency_detail_top_k,
                        "num_annotation_edges": detail.num_annotation_edges,
                        "n_queries": detail.n_queries,
                        "n_samples": detail.n_samples,
                    },
                    "query_stats": detail.query_stats,
                    "edge_stats": detail.edge_stats,
                }, ensure_ascii=False) + "\n")

        if self.state.global_step % self.args.logging_steps == 0:
            log_payload = {
                "ntp_loss": ntp_loss.item(),
                "saliency_loss": saliency_loss.item(),
                "saliency_loss_type_id": 1.0 if self.saliency_loss_type == "softmax" else 0.0,
                "total_loss": total_loss.item(),
            }
            if self.loss_mode != "ce_only":
                log_payload.update({
                    "saliency_eps_num": self.saliency_eps,
                    "saliency_eps_floor_effective": diag.floor_eps if diag is not None else self.saliency_floor_eps,
                    "saliency_eps_floor_batch": diag.batch_floor_eps if diag is not None else 0.0,
                    "saliency_eps_floor_logit": diag.floor_logit_eps if diag is not None else 0.0,
                })
            if detail is not None:
                log_payload.update({
                    "strict_Cbar": detail.avg_C,
                    "strict_Nbar": detail.avg_N,
                    "strict_ratio": detail.avg_ratio,
                    "strict_active_margin_rate": detail.active_margin_rate,
                    "strict_hit_at_k": detail.hit_at_k,
                    "strict_recall_at_k": detail.recall_at_k,
                    "strict_precision_at_k": detail.precision_at_k,
                    "strict_map_at_k": detail.map_at_k,
                })
            self.log(log_payload)

        return (total_loss, outputs) if return_outputs else total_loss




@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    use_flash_attention: bool = field(default=False)
    use_peft: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    data_path: str = field(default="./data/mceval/mceval-sft.jsonl")
    max_len: int = field(default=8192)
    language: Optional[Literal["Python", "C#", "CPP", "Go", "Java", "C"]] = field(
        default=None,
        metadata={"help": "Filter to one language: Python | C# | CPP | Go | Java | C. None = all."}
    )


@dataclass
class SFTTrainingArguments(TrainingArguments):
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "When resuming, load model/trainer state but skip optimizer/scheduler .pt files. Useful with torch<2.6 safety restrictions."}
    )
    saliency_lambda: float = field(
        default=0.1,
        metadata={"help": "Weight γ in L_total = L_next + γ·L_contrib."}
    )
    saliency_alpha: float = field(
        default=1.5,
        metadata={"help": "Temperature tau for saliency logits. softmax_margin uses log(C + eps) / tau; softmax uses C / tau."}
    )
    saliency_eps: float = field(
        default=1e-8,
        metadata={"help": "Numerical epsilon for log(C + eps), safe division, and diagnostics."}
    )
    saliency_floor_eps: float = field(
        default=0.0,
        metadata={"help": "Fixed saliency-space floor for negative logits when saliency_floor_eps_mode=fixed."}
    )
    saliency_floor_eps_mode: str = field(
        default="ema_quantile",
        metadata={"help": "Floor eps schedule: fixed | batch_quantile | ema_quantile."}
    )
    saliency_floor_quantile: float = field(
        default=0.75,
        metadata={"help": "Quantile over current causal-source saliency values for dynamic floor eps."}
    )
    saliency_floor_ema_beta: float = field(
        default=0.95,
        metadata={"help": "EMA beta for saliency_floor_eps_mode=ema_quantile."}
    )
    saliency_floor_min_eps: float = field(
        default=1e-8,
        metadata={"help": "Minimum positive eps used by dynamic floor schedules."}
    )
    saliency_floor_warmup_steps: int = field(
        default=10,
        metadata={"help": "Use floor_eps=0 for the first N dynamic-floor saliency steps; N=10 by default."}
    )
    saliency_floor_logit_eps: Optional[float] = field(
        default=None,
        metadata={"help": "If set, softmax_margin uses this fixed logit-space negative floor: max(l_neg, saliency_floor_logit_eps). Example: -10."}
    )
    saliency_loss_type: str = field(
        default="softmax_margin",
        metadata={"help": "Saliency objective: softmax_margin log(C+eps)/tau with negative floor, or softmax raw C/tau over all causal sources. Legacy alias infonce_floor is accepted."}
    )
    saliency_margin_plus: float = field(
        default=2.08,
        metadata={"help": "[margin_bce] Lower-bound margin on log(C+eps) for annotated positives (default log(8))."}
    )
    saliency_margin_minus: float = field(
        default=0.41,
        metadata={"help": "[margin_bce] Upper-bound margin on log(C+eps) for non-annotated negatives; no penalty once r <= m_minus (default log(1.5))."}
    )
    saliency_margin_gamma: float = field(
        default=2.0,
        metadata={"help": "[margin_bce] Sharpness of the per-edge softplus (sigmoid gradient slope)."}
    )
    saliency_neg_weight: float = field(
        default=0.5,
        metadata={"help": "[margin_bce] Multiplier for the negative-side mean penalty."}
    )
    saliency_neg_hard_only: bool = field(
        default=False,
        metadata={"help": "[margin_bce] If True, the negative-side mean only averages over negatives with r > m_minus."}
    )
    saliency_neg_sample_k: int = field(
        default=0,
        metadata={"help": "[softmax_margin/softmax/ranknet/contrastive] If >0, sample this many negatives per query for the saliency loss (MoCo-style). 0 = use all causal negatives (default)."}
    )
    saliency_source_chunk_size: int = field(
        default=16,
        metadata={"help": "Source-token chunk size for saliency contribution rows. Lower values reduce peak memory; use 4 or 2 for long/high-edge samples."}
    )
    loss_mode: str = field(
        default="ce_saliency",
        metadata={"help": "Training objective: ce_saliency | saliency_only | ce_only."}
    )
    saliency_detail_log_path: str = field(
        default="",
        metadata={"help": "Optional JSONL path for strict-causal query/edge saliency diagnostics."}
    )
    saliency_detail_log_steps: int = field(
        default=0,
        metadata={"help": "Write detailed saliency diagnostics every N global steps; 0 disables."}
    )
    saliency_detail_top_k: int = field(
        default=10,
        metadata={"help": "Top-k threshold for annotation hit@k in detailed diagnostics."}
    )
    enable_attn_viz: bool = field(
        default=False,
        metadata={"help": "Capture step-0 attention visualization samples during training. Disabled by default to save memory."}
    )
    attn_viz_num_samples: int = field(
        default=10,
        metadata={"help": "Number of samples for step-0 attention visualization when enable_attn_viz=True."}
    )
    cache_dir: Optional[str] = field(default=None)




def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, SFTTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.saliency_loss_type = canonical_saliency_loss_type(training_args.saliency_loss_type)
    # Keep custom fields such as annot_pairs. HF Trainer otherwise prunes keys
    # that are not in model.forward(), which silently disables saliency loss.
    training_args.remove_unused_columns = False

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        cache_dir=training_args.cache_dir,
        model_max_length=data_args.max_len,
        truncation=True,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|im_end|>", "<|im_start|>"]
    })

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset = AnnotatedSFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_len=data_args.max_len,
        language=data_args.language,
    )
    data_collator = DataCollatorForAnnotatedSFT(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_args.use_flash_attention:
        logger.warning(
            "use_flash_attention=True is incompatible with the saliency loss "
            "(needs eager attention to expose A^h_{i,j}). Falling back to eager."
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if model_args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # ── Trainer ───────────────────────────────────────────────────────────────
    callbacks = []
    if training_args.enable_attn_viz:
        callbacks.append(AttentionVisualizationCallback(
            dataset=train_dataset,
            tokenizer=tokenizer,
            output_dir=os.path.join(training_args.output_dir, "attn_viz"),
            num_samples=training_args.attn_viz_num_samples,
        ))

    trainer = AnnotatedSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        eval_dataset=train_dataset,
        train_dataset=train_dataset,
        data_collator=data_collator,
        saliency_lambda=training_args.saliency_lambda,
        saliency_alpha=training_args.saliency_alpha,
        saliency_eps=training_args.saliency_eps,
        saliency_floor_eps=training_args.saliency_floor_eps,
        saliency_floor_eps_mode=training_args.saliency_floor_eps_mode,
        saliency_floor_quantile=training_args.saliency_floor_quantile,
        saliency_floor_ema_beta=training_args.saliency_floor_ema_beta,
        saliency_floor_min_eps=training_args.saliency_floor_min_eps,
        saliency_floor_warmup_steps=training_args.saliency_floor_warmup_steps,
        saliency_floor_logit_eps=training_args.saliency_floor_logit_eps,
        saliency_loss_type=training_args.saliency_loss_type,
        loss_mode=training_args.loss_mode,
        saliency_detail_log_path=training_args.saliency_detail_log_path,
        saliency_detail_log_steps=training_args.saliency_detail_log_steps,
        saliency_detail_top_k=training_args.saliency_detail_top_k,
        saliency_margin_plus=training_args.saliency_margin_plus,
        saliency_margin_minus=training_args.saliency_margin_minus,
        saliency_margin_gamma=training_args.saliency_margin_gamma,
        saliency_neg_weight=training_args.saliency_neg_weight,
        saliency_neg_hard_only=training_args.saliency_neg_hard_only,
        saliency_neg_sample_k=training_args.saliency_neg_sample_k,
        saliency_source_chunk_size=training_args.saliency_source_chunk_size,
        callbacks=callbacks,
    )

    logger.info(
        "Training objective: loss_mode=%s saliency_loss=%s saliency_loss_type=%s "
        "saliency_lambda=%s tau=%s eps_num=%s floor_eps=%s floor_logit_eps=%s floor_mode=%s "
        "floor_quantile=%s floor_warmup=%s source_chunk=%s neg_sample_k=%s output_dir=%s",
        training_args.loss_mode,
        saliency_loss_display_name(training_args.saliency_loss_type),
        training_args.saliency_loss_type,
        training_args.saliency_lambda,
        training_args.saliency_alpha,
        training_args.saliency_eps,
        training_args.saliency_floor_eps,
        training_args.saliency_floor_logit_eps,
        training_args.saliency_floor_eps_mode,
        training_args.saliency_floor_quantile,
        training_args.saliency_floor_warmup_steps,
        training_args.saliency_source_chunk_size,
        training_args.saliency_neg_sample_k,
        training_args.output_dir,
    )
    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt:
        logger.info("Resuming Trainer state from checkpoint: %s", resume_ckpt)
        if getattr(training_args, "resume_model_only", False):
            logger.warning(
                "resume_model_only=True: skipping optimizer/scheduler restore to avoid torch.load safety restriction; "
                "model weights and Trainer state will still be loaded from the checkpoint."
            )
            def _skip_optimizer_and_scheduler(resume_from_checkpoint):
                logger.warning("Skipped optimizer/scheduler restore from %s", resume_from_checkpoint)
                return None
            trainer._load_optimizer_and_scheduler = _skip_optimizer_and_scheduler
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        objective = {
            "loss_mode": training_args.loss_mode,
            "saliency_loss_type": training_args.saliency_loss_type,
            "saliency_loss_name": saliency_loss_display_name(training_args.saliency_loss_type),
            "saliency_lambda": training_args.saliency_lambda,
            "saliency_temperature_tau": training_args.saliency_alpha,
            "saliency_eps_num": training_args.saliency_eps,
            "saliency_floor_eps": training_args.saliency_floor_eps,
            "saliency_floor_logit_eps": training_args.saliency_floor_logit_eps,
            "saliency_floor_eps_mode": training_args.saliency_floor_eps_mode,
            "saliency_floor_quantile": training_args.saliency_floor_quantile,
            "saliency_floor_ema_beta": training_args.saliency_floor_ema_beta,
            "saliency_floor_min_eps": training_args.saliency_floor_min_eps,
            "saliency_floor_warmup_steps": training_args.saliency_floor_warmup_steps,
            "saliency_source_chunk_size": training_args.saliency_source_chunk_size,
            "saliency_neg_sample_k": training_args.saliency_neg_sample_k,
            "data_path": data_args.data_path,
            "model_name_or_path": model_args.model_name_or_path,
            "run_name": training_args.run_name,
        }
        with open(os.path.join(training_args.output_dir, "saliency_training_config.json"), "w", encoding="utf-8") as f:
            json.dump(objective, f, ensure_ascii=False, indent=2)
        readme_path = os.path.join(training_args.output_dir, "README.md")
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write(
                "\n## GraphSignal Training Objective\n\n"
                f"- Loss mode: `{training_args.loss_mode}`\n"
                f"- Saliency loss: `{objective['saliency_loss_name']}` (`{training_args.saliency_loss_type}`)\n"
                f"- Saliency lambda: `{training_args.saliency_lambda}`\n"
                f"- Saliency temperature tau: `{training_args.saliency_alpha}`\n"
                f"- Numerical epsilon: `{training_args.saliency_eps}`\n"
                f"- Floor epsilon mode: `{training_args.saliency_floor_eps_mode}`\n"
                f"- Floor epsilon fixed value: `{training_args.saliency_floor_eps}`\n"
                f"- Floor logit epsilon: `{training_args.saliency_floor_logit_eps}`\n"
                f"- Floor epsilon quantile: `{training_args.saliency_floor_quantile}`\n"
                f"- Floor epsilon warmup steps: `{training_args.saliency_floor_warmup_steps}`\n"
                f"- Saliency negative sample K: `{training_args.saliency_neg_sample_k}`\n"
                f"- Training data: `{data_args.data_path}`\n"
            )


if __name__ == "__main__":
    train()
