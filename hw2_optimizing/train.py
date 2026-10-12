import json
import math
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

if Path("/usr/local/cuda/bin/ptxas").is_file():
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.utils import ModelOutput
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


DATA: Path = Path("data")
TRAIN_SECONDS: float = 300
ATTENTION_IMPLEMENTATION: str = "sdpa"
COMPILE_FUSED_LOSS_EAGER: bool = False

MODEL_CONFIG: dict[str, Any] = dict(
    hidden_size=2048,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=8,
    intermediate_size=8192,
    head_dim=128,
    use_cache=False,
)

TRAINING_CONFIG: dict[str, Any] = dict(
    output_dir="results/baseline",
    # Настройки для экспериментов.
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    torch_compile=False,
    use_liger_kernel=False,
    activation_offloading=False,
    packing=False,
    padding_free=False,
    # Общие параметры обучения и замера.
    max_steps=1_000_000,
    learning_rate=3e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=200,
    weight_decay=0.01,
    optim="adamw_torch_fused",
    bf16=True,
    tf32=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing_strategy="bfd",
    max_length=512,
    completion_only_loss=False,
    loss_type="nll",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    logging_steps=10,
    logging_nan_inf_filter=False,
    report_to="none",
    save_strategy="no",
    eval_strategy="no",
    seed=42,
    data_seed=42,
    disable_tqdm=True,
)

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": dict(
        output_dir="results/baseline",
    ),
    "bs64_ga1": dict(
        output_dir="results/bs64_ga1",
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
    ),
    "packing": dict(
        output_dir="results/packing",
        packing=True,
    ),
    "padding_free": dict(
        output_dir="results/padding_free",
        padding_free=True,
    ),
    "grad_ckpt_bs64": dict(
        output_dir="results/grad_ckpt_bs64",
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
    ),
    "compile": dict(
        output_dir="results/compile",
        torch_compile=True,
    ),
    "flash_attention_2": dict(
        output_dir="results/flash_attention_2",
        attention_implementation="flash_attention_2",
    ),
    "liger": dict(
        output_dir="results/liger",
        use_liger_kernel=True,
    ),
    "offloading_ckpt_bs64": dict(
        output_dir="results/offloading_ckpt_bs64",
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        activation_offloading=True,
    ),
    "packing_padding_free": dict(
        output_dir="results/packing_padding_free",
        packing=True,
        padding_free=True,
    ),
    "best_candidate": dict(
        output_dir="results/best_candidate",
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        packing=True,
        padding_free=True,
    ),
}

# BFD packing uses flattened batches and needs sequence-boundary-aware attention.
for name in ("packing", "padding_free", "packing_padding_free", "best_candidate"):
    EXPERIMENTS[name]["attention_implementation"] = "flash_attention_2"

EXPERIMENTS.update({
    "bs16_ga4": dict(per_device_train_batch_size=16, gradient_accumulation_steps=4),
    "checkpointing": dict(gradient_checkpointing=True),
    "offloading": dict(gradient_checkpointing=True, activation_offloading=True),
    "offloading_only": dict(gradient_checkpointing=False, activation_offloading=True),
    "offloading_only_fa2": dict(gradient_checkpointing=False, activation_offloading=True,
                               attention_implementation="flash_attention_2"),
    "fa2_bs64": dict(attention_implementation="flash_attention_2",
                     per_device_train_batch_size=64, gradient_accumulation_steps=1),
    "pack_liger": dict(attention_implementation="flash_attention_2",
                       packing=True, use_liger_kernel=True),
    "pack_compile": dict(attention_implementation="flash_attention_2",
                         packing=True, torch_compile=True),
    "pack_liger_compile": dict(attention_implementation="flash_attention_2",
                               packing=True, use_liger_kernel=True, torch_compile=True),
    "pack_liger_compile_eager_loss": dict(attention_implementation="flash_attention_2",
        packing=True, use_liger_kernel=True, torch_compile=True,
        compile_fused_loss_eager=True),
    "pack_bs16": dict(attention_implementation="flash_attention_2", packing=True,
                      per_device_train_batch_size=16, gradient_accumulation_steps=4),
    "pack_liger_bs64": dict(attention_implementation="flash_attention_2", packing=True,
                            use_liger_kernel=True, per_device_train_batch_size=64,
                            gradient_accumulation_steps=1),
    "fa2_bs96": dict(attention_implementation="flash_attention_2",
                     per_device_train_batch_size=96, gradient_accumulation_steps=1),
    "fa2_bs128": dict(attention_implementation="flash_attention_2",
                      per_device_train_batch_size=128, gradient_accumulation_steps=1),
    "pf_bs64": dict(attention_implementation="flash_attention_2", padding_free=True,
                    per_device_train_batch_size=64, gradient_accumulation_steps=1),
    "pf_bs128": dict(attention_implementation="flash_attention_2", padding_free=True,
                     per_device_train_batch_size=128, gradient_accumulation_steps=1),
    "pf_liger": dict(attention_implementation="flash_attention_2", padding_free=True,
                     use_liger_kernel=True),
    "pf_liger_bs64": dict(attention_implementation="flash_attention_2", padding_free=True,
                          use_liger_kernel=True, per_device_train_batch_size=64,
                          gradient_accumulation_steps=1),
    "pf_compile": dict(attention_implementation="flash_attention_2", padding_free=True,
                       torch_compile=True),
    "pack_liger_noce_compile": dict(attention_implementation="flash_attention_2",
        packing=True, use_liger_kernel=True, torch_compile=True,
        liger_kernel_config={"fused_linear_cross_entropy": False}),
    "pack_liger_noce_bs64_compile": dict(attention_implementation="flash_attention_2",
        packing=True, use_liger_kernel=True, torch_compile=True,
        per_device_train_batch_size=64, gradient_accumulation_steps=1,
        liger_kernel_config={"fused_linear_cross_entropy": False}),
    "pack_liger_noce_ga1_compile": dict(attention_implementation="flash_attention_2",
        packing=True, use_liger_kernel=True, torch_compile=True,
        gradient_accumulation_steps=1,
        liger_kernel_config={"fused_linear_cross_entropy": False}),
})

EXPERIMENTS["best_candidate"] = dict(EXPERIMENTS["pack_liger_noce_bs64_compile"])
EXPERIMENTS["best_candidate"]["output_dir"] = "results/final_best_candidate"


def configure_from_cli() -> None:
    global ATTENTION_IMPLEMENTATION, TRAIN_SECONDS, COMPILE_FUSED_LOSS_EAGER

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=os.environ.get("EXPERIMENT", "best_candidate"),
    )
    parser.add_argument(
        "--train-seconds",
        type=float,
        default=float(os.environ.get("TRAIN_SECONDS", TRAIN_SECONDS)),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR"),
        help="Override TRAINING_CONFIG['output_dir'] for one-off runs.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=os.environ.get("ATTENTION_IMPLEMENTATION", ATTENTION_IMPLEMENTATION),
    )
    args = parser.parse_args()

    TRAIN_SECONDS = args.train_seconds
    experiment_config = dict(EXPERIMENTS[args.experiment])
    COMPILE_FUSED_LOSS_EAGER = experiment_config.pop("compile_fused_loss_eager", False)
    ATTENTION_IMPLEMENTATION = experiment_config.pop(
        "attention_implementation", args.attn_implementation
    )
    TRAINING_CONFIG.update(experiment_config)
    if args.output_dir is not None:
        TRAINING_CONFIG["output_dir"] = args.output_dir
    TRAINING_CONFIG["run_name"] = args.experiment


def prepare_data() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "ai-forever/rugpt3small_based_on_gpt2",
        revision="a9307e696cd3c5b7f953ff4cb19d76a4d81821d5",
    )
    tokenizer.pad_token = tokenizer.eos_token
    source = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ru",
        split="train",
        streaming=True,
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    )
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for index, article in enumerate(source.take(50_500)):
        split = "validation" if index < 500 else "train"
        paragraphs = [
            p.strip() for p in article["text"].splitlines() if len(p.strip()) >= 80
        ][:8]
        if paragraphs:
            encoded = tokenizer(
                paragraphs,
                add_special_tokens=False,
                truncation=True,
                max_length=511,
                return_attention_mask=False,
            )["input_ids"]
            for paragraph_index, tokens in enumerate(encoded):
                if len(tokens) >= 32:
                    rows[split].append(
                        dict(
                            input_ids=tokens + [tokenizer.eos_token_id],
                            article_id=article["id"],
                            paragraph_index=paragraph_index,
                        )
                    )
        if (index + 1) % 5000 == 0:
            print(f"Подготовлено {index + 1}/50500 статей", flush=True)
    DatasetDict(
        {name: Dataset.from_list(items) for name, items in rows.items()}
    ).save_to_disk(str(DATA / "dataset"))
    tokenizer.save_pretrained(DATA / "tokenizer")


class CausalCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().torch_call(examples)
        batch["labels"][:, 0] = -100
        return batch


class BenchmarkTrainer(SFTTrainer):
    target_tokens: torch.Tensor

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelOutput]:
        return Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.target_tokens += inputs["labels"].ne(-100).sum()
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        for key in ("loss", "grad_norm", "train_loss"):
            if key in logs and not math.isfinite(logs[key]):
                raise FloatingPointError(f"Значение {key} не валидное")
        logs.pop("train_samples_per_second", None)
        logs.pop("train_steps_per_second", None)
        super().log(logs, start_time)


class Benchmark(TrainerCallback):
    start: float
    previous_end: float
    previous_tokens: int

    def __init__(self, trainer: BenchmarkTrainer, seconds: float) -> None:
        self.trainer, self.seconds = trainer, seconds
        self.steps: list[dict[str, float | int | bool]] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = self.previous_end = time.perf_counter()
        self.previous_tokens = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        tokens = self.trainer.target_tokens.item()
        record: dict[str, float | int | bool] = dict(
            step=state.global_step,
            elapsed_seconds=now - self.start,
            step_seconds=now - self.previous_end,
            target_tokens=tokens - self.previous_tokens,
            warmup=state.global_step <= 10,
        )
        self.steps.append(record)
        if state.global_step % 10 == 0:
            print(json.dumps(record), flush=True)
        self.previous_end, self.previous_tokens = now, tokens
        if now - self.start >= self.seconds:
            control.should_training_stop = True

    def summary(self) -> dict[str, float | int | None]:
        measured = self.steps[10:]
        duration = sum(step["step_seconds"] for step in measured)
        tokens = sum(step["target_tokens"] for step in measured)
        elapsed = self.previous_end - self.start
        return dict(
            train_seconds=elapsed,
            optimizer_steps=len(self.steps),
            target_tokens=self.previous_tokens,
            tokens_per_second_including_warmup=self.previous_tokens / elapsed,
            warmup_steps=10,
            measured_steps=len(measured),
            measured_seconds=duration,
            tokens_per_second=tokens / duration if measured else None,
            seconds_per_optimizer_step=duration / len(measured) if measured else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset.select_columns(["input_ids"]),
        batch_size=8,
        collate_fn=CausalCollator(tokenizer.pad_token_id, completion_only_loss=False),
    )
    model.eval()
    loss_sum, tokens = 0.0, 0
    for batch in loader:
        batch = {key: value.cuda() for key, value in batch.items()}
        count = batch["labels"].ne(-100).sum().item()
        loss_sum += model(**batch).loss.item() * count
        tokens += count
    loss = loss_sum / tokens

    return dict(validation_loss=loss, validation_target_tokens=tokens)


def main() -> None:
    configure_from_cli()
    if COMPILE_FUSED_LOSS_EAGER:
        import liger_kernel.transformers.model.qwen3 as liger_qwen3

        # Keep the fused Triton loss, but exclude its custom autograd from Dynamo.
        liger_qwen3.LigerForCausalLMLoss = torch.compiler.disable(
            liger_qwen3.LigerForCausalLMLoss
        )
    training_args = SFTConfig(**TRAINING_CONFIG)

    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(dict(
        training_config=TRAINING_CONFIG, model_config=MODEL_CONFIG,
        attention_implementation=ATTENTION_IMPLEMENTATION,
        requested_train_seconds=TRAIN_SECONDS,
        compile_fused_loss_eager=COMPILE_FUSED_LOSS_EAGER,
    ), indent=2) + "\n")
    if not (DATA / "dataset").exists():
        prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(DATA / "tokenizer")
    dataset = load_from_disk(str(DATA / "dataset"))

    set_seed(training_args.seed)
    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG,
    )
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        torch_dtype=torch.bfloat16,
    )

    trainer = BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        data_collator=(
            None
            if training_args.packing or training_args.padding_free
            else CausalCollator(tokenizer.pad_token_id)
        ),
    )
    trainer.target_tokens = torch.zeros((), dtype=torch.int64, device="cuda")
    benchmark = Benchmark(trainer, TRAIN_SECONDS)
    trainer.add_callback(benchmark)

    trainer.train()
    summary = benchmark.summary()
    summary.update(
        evaluate(trainer.model, dataset["validation"].select(range(512)), tokenizer)
    )
    summary["attention_implementation"] = ATTENTION_IMPLEMENTATION
    summary["training_config"] = TRAINING_CONFIG
    summary["model_config"] = MODEL_CONFIG
    summary["compile_fused_loss_eager"] = COMPILE_FUSED_LOSS_EAGER
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
