"""Run one independent Full FT/LoRA experiment and evaluate frozen test data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from prepare_data import dump


class Collator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, features):
        length = max(len(f["input_ids"]) for f in features)
        inputs, labels, masks = [], [], []
        for f in features:
            padding = length - len(f["input_ids"])
            inputs.append(f["input_ids"] + [self.pad] * padding)
            labels.append(f["labels"] + [-100] * padding)
            masks.append([1] * len(f["input_ids"]) + [0] * padding)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in
                {"input_ids": inputs, "labels": labels, "attention_mask": masks}.items()}


class MemoryMonitor:
    """Sample device memory every second, in addition to PyTorch peak counters."""
    def __init__(self, gpu):
        self.gpu, self.peak, self.samples, self.errors = gpu, 0, 0, 0
        self.stop = threading.Event()

    def sample(self):
        try:
            value = subprocess.check_output([
                "nvidia-smi", f"--id={self.gpu}", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"], text=True, timeout=5)
            self.peak = max(self.peak, int(value.strip()))
            self.samples += 1
        except (ValueError, subprocess.SubprocessError, OSError):
            self.errors += 1

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(1)

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=10)
        self.sample()


def read_split(name, manifest):
    path = Path("data") / f"{name}.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["splits"][name]["file_sha256"]
    return [json.loads(line) for line in path.read_text().splitlines()]


def train_dataset(rows):
    return Dataset.from_list([{k: r[k] for k in ["input_ids", "labels"]} for r in rows])


def evaluate(model, tokenizer, rows, generation, out):
    scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)
    tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    results = []
    started = time.perf_counter()
    # Each line is flushed so progress and partial predictions survive interruption.
    with (out / "predictions.jsonl").open("w") as f:
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            inputs = tokenizer([r["prompt"] for r in batch], add_special_tokens=False,
                               padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                tokens = model.generate(**inputs, **generation,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        temperature=None, top_p=None, top_k=None)
            predictions = tokenizer.batch_decode(tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, prediction in zip(batch, predictions):
                scores = scorer.score(row["reference"], prediction.strip())
                result = {"id": row["id"], "reference": row["reference"], "prediction": prediction.strip(),
                          "rouge2": scores["rouge2"].fmeasure * 100,
                          "rougeL": scores["rougeL"].fmeasure * 100}
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"EVAL {len(results)}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    return {"rouge2": float(np.mean([r["rouge2"] for r in results])),
            "rougeL": float(np.mean([r["rougeL"] for r in results])),
            "test_examples": len(rows), "evaluation_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("This VM reservation requires CUDA_VISIBLE_DEVICES=1")
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    config = json.loads(Path("config.json").read_text())
    manifest = json.loads(Path("data/manifest.json").read_text())
    assert config == manifest["config"], "Freeze configuration before preparing data"
    revisions = manifest["revisions"]
    experiment = config["experiments"][args.experiment]
    out = Path("runs") / args.experiment
    out.mkdir(parents=True, exist_ok=False)
    set_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revisions["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=revisions["model"],
                                                dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    lora = None
    if experiment["method"] == "lora":
        lora = {**config["lora_defaults"], **experiment.get("overrides", {})}
        model = get_peft_model(model, LoraConfig(**lora))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {"experiment": args.experiment, "method": experiment["method"], "lora": lora,
               "total_parameters": total,
               "trainable_parameters": 0 if experiment["method"] == "base" else trainable,
               "seed": config["seed"], "revisions": revisions,
               "gpu": torch.cuda.get_device_name(), "physical_gpu": 1,
               "torch": torch.__version__, "config": config}
    dump(out / "configuration.json", summary)
    if experiment["method"] != "base":
        train = read_split("train", manifest)
        validation = read_split("validation", manifest)
        model.config.use_cache = False
        trainer = Trainer(model=model,
                          args=TrainingArguments(output_dir=str(out / "checkpoints"), label_names=["labels"],
                                                 seed=config["seed"], data_seed=config["seed"],
                                                 **config["training"]),
                          train_dataset=train_dataset(train), eval_dataset=train_dataset(validation),
                          data_collator=Collator(tokenizer.pad_token_id), processing_class=tokenizer)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with MemoryMonitor("1") as monitor:
            result = trainer.train()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        summary.update({"training_seconds_including_validation_and_checkpoints": elapsed,
                        "train_metrics": result.metrics, "optimizer_steps": trainer.state.global_step,
                        "best_validation_loss": trainer.state.best_metric,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        "sampled_device_peak_mib": monitor.peak if monitor.samples else None,
                        "memory_samples": monitor.samples, "memory_sampling_errors": monitor.errors})
        trainer.save_model(str(out / "model"))
        tokenizer.save_pretrained(out / "model")
        dump(out / "training_history.json", trainer.state.log_history)
        dump(out / "training_summary.json", summary)
        del trainer, train, validation
    else:
        summary["training_seconds_including_validation_and_checkpoints"] = 0
    summary.update(evaluate(model, tokenizer, read_split("test", manifest), config["generation"], out))
    dump(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["experiment", "rouge2", "rougeL", "test_examples"]}), flush=True)


if __name__ == "__main__":
    main()
