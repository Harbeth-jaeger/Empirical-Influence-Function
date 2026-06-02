from __future__ import annotations

import argparse
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import transformers
from peft import PeftConfig, get_peft_model
from transformers import Trainer

from scripts.benchmark.ibft_loss import VariationalBottleneck, compute_ibft_loss
from src.sft.binarize_data import setup_tokenizer
from src.sft.train import (
    DataArguments,
    LoggingCallback,
    ModelArguments,
    SaveModelCallback,
    TrainingArguments,
    find_latest_checkpoint,
    make_supervised_data_module,
)


@dataclass
class IBFTTrainingArguments(TrainingArguments):
    ib_layer: int = field(
        default=20,
        metadata={"help": "Hidden-state tensor index used by the IB bottleneck. Negative values count from the end."},
    )
    ib_z_dim: int = field(default=256, metadata={"help": "Bottleneck latent dimension."})
    ib_alpha: float = field(default=0.01, metadata={"help": "Weight of the IB auxiliary loss."})
    ib_beta: float = field(default=1.0, metadata={"help": "Weight of bottleneck prediction CE inside the IB loss."})
    ib_max_tokens_per_batch: int = field(
        default=2048,
        metadata={"help": "Maximum supervised token positions sampled per micro-batch for IB loss."},
    )
    ib_dropout: float = field(default=0.0, metadata={"help": "Dropout in the variational bottleneck encoder."})


class IBFTTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_ib_log_step = -1

    def _module_with_ibft(self, model: torch.nn.Module) -> torch.nn.Module:
        for candidate in (self.model, model, getattr(model, "module", None)):
            if candidate is not None and getattr(candidate, "ib_bottleneck", None) is not None:
                return candidate
        raise RuntimeError("IBFTTrainer requires model.ib_bottleneck to be attached before training.")

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        outputs = model(**inputs, output_hidden_states=self.args.ib_alpha > 0)
        ce_loss = outputs.loss

        if self.args.ib_alpha <= 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        ibft_model = self._module_with_ibft(model)
        bottleneck = ibft_model.ib_bottleneck
        lm_head = ibft_model.get_output_embeddings()
        ib = compute_ibft_loss(
            hidden_states=outputs.hidden_states,
            labels=inputs["labels"],
            lm_head=lm_head,
            bottleneck=bottleneck,
            layer=self.args.ib_layer,
            beta=self.args.ib_beta,
            max_tokens=self.args.ib_max_tokens_per_batch,
        )
        loss = ce_loss + self.args.ib_alpha * ib.loss

        logging_steps = max(int(getattr(self.args, "logging_steps", 0) or 0), 1)
        if self.state.global_step != self._last_ib_log_step and self.state.global_step % logging_steps == 0:
            self._last_ib_log_step = self.state.global_step
            self.log(
                {
                    "ce_loss": float(ce_loss.detach().float().cpu()),
                    "ib_loss": float(ib.loss.detach().float().cpu()),
                    "ib_kl_loss": float(ib.kl_loss.detach().float().cpu()),
                    "ib_z_ce_loss": float(ib.z_ce_loss.detach().float().cpu()),
                    "ib_tokens": ib.num_tokens,
                }
            )

        return (loss, outputs) if return_outputs else loss


def _is_distributed(training_args: transformers.TrainingArguments) -> bool:
    return (
        training_args.local_rank != -1
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
        or "LOCAL_RANK" in os.environ
        or "RANK" in os.environ
    )


def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, IBFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    visible_gpus = torch.cuda.device_count()
    if visible_gpus > 1 and (not _is_distributed(training_args)) and training_args.per_device_train_batch_size < visible_gpus:
        raise ValueError(
            "Invalid multi-GPU config: per_device_train_batch_size "
            f"({training_args.per_device_train_batch_size}) < visible GPU count ({visible_gpus}). "
            "Use torchrun for DDP, fewer GPUs, or a larger per-device batch size."
        )

    merged_args = argparse.Namespace(**{**model_args.__dict__, **data_args.__dict__, **training_args.__dict__})

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else "sdpa",
        trust_remote_code=True,
        local_files_only=True,
    )
    if training_args.use_peft:
        peft_config = PeftConfig.from_pretrained(training_args.peft_config_path)
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
    if hidden_size <= 0:
        raise ValueError("Unable to infer model hidden size for IB-FT bottleneck.")
    model.ib_bottleneck = VariationalBottleneck(
        hidden_size=hidden_size,
        z_dim=training_args.ib_z_dim,
        dropout=training_args.ib_dropout,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        cache_dir=None,
        model_max_length=training_args.model_max_length,
        truncation=True,
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
    )
    old_vocab_size = len(tokenizer)
    tokenizer = setup_tokenizer(tokenizer)
    if len(tokenizer) != old_vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    data_module = make_supervised_data_module(tokenizer=tokenizer, args=merged_args)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        **data_module,
        callbacks=[LoggingCallback, SaveModelCallback],
    )
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = IBFTTrainer(**trainer_kwargs)

    latest_checkpoint = find_latest_checkpoint(training_args.output_dir)
    resume_enabled = os.environ.get("EIF_RESUME_CHECKPOINT", "0").strip().lower() in {"1", "true", "yes", "y"}
    if latest_checkpoint and resume_enabled:
        print(f"checkpoint found, resume training from: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        if latest_checkpoint:
            print(
                "checkpoint found, but resume is disabled (set EIF_RESUME_CHECKPOINT=1 to enable). "
                "Starting a fresh training run."
            )
        trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
