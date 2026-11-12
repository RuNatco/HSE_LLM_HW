import argparse
import json
import os
import shutil
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from common import ROOT, MODEL, REVISION, read_json, read_jsonl, write_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["pilot", "rl1_grpo_strict", "rl2_grpo_shaped", "rl2_grpo_shaped_90", "rl3_drgrpo_shaped"])
    p.add_argument("--config", default="config.json")
    args = p.parse_args()
    settings = read_json(args.config)
    experiment = settings["experiments"][args.experiment]
    cfg = {**settings["common"], **experiment}
    cfg["training"] = {**settings["common"]["training"], **experiment["training"]}
    if cfg["model"] != MODEL or cfg["revision"] != REVISION:
        raise SystemExit("Model/revision must match the assignment evaluator")
    output = ROOT / cfg["output_dir"]
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output already exists: {output}. Use a new run directory.")
    if shutil.disk_usage(ROOT).free < cfg["min_free_disk_gib"] * 2**30:
        raise SystemExit("Insufficient free disk space")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from rewards import RewardEngine, make_reward
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one reserved GPU using CUDA_VISIBLE_DEVICES / Docker --gpus")
    output.mkdir(parents=True)
    write_json(output / "config.json", cfg)
    versions = {name: version(name) for name in ["torch", "transformers", "trl", "peft", "vllm", "datasets"]}
    write_json(output / "environment.json", {"versions": versions,
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv"], text=True)})
    data_dir = ROOT / cfg["data_dir"]
    manifest = read_json(data_dir / "manifest.json")
    for name, digest in manifest["files"].items():
        if sha256(data_dir / name) != digest:
            raise RuntimeError(f"Prepared data changed: {name}")
    write_json(output / "data_manifest.json", manifest)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")[:cfg["validation_examples"]]
    if cfg.get("train_examples"):
        train_rows = train_rows[:cfg["train_examples"]]

    write_json(output / "validation_ids.json", [row["id"] for row in val_rows])
    set_seed(cfg["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa")
    suffixes = set(cfg["lora"]["target_modules"])
    targets = [name for name, module in model.named_modules()
               if "language_model" in name and name.rsplit(".", 1)[-1] in suffixes
               and isinstance(module, torch.nn.Linear)]
    write_json(output / "lora_targets.json", targets)
    lora = LoraConfig(**{**cfg["lora"], "target_modules": targets})
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    engine = RewardEngine(tokenizer)
    started = time.monotonic()

    class Validation(TrainerCallback):
        best = -1.0
        best_step = None
        last_step = None

        def evaluate(self, state, model):
            if self.last_step == state.global_step:
                return
            self.last_step = state.global_step
            was_training = model.training
            model.eval()
            had_checkpointing = model.is_gradient_checkpointing
            if had_checkpointing:
                model.gradient_checkpointing_disable()
            results = []
            try:
                with torch.inference_mode():
                    for row in val_rows:
                        inputs = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
                        generated = model.generate(**inputs, do_sample=False, max_new_tokens=cfg["validation_max_new_tokens"],
                            use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=list(eos_ids))
                        ids = generated[0, inputs.input_ids.shape[1]:].tolist()
                        text = tokenizer.decode(ids, skip_special_tokens=False)
                        score = engine.score(text, json.loads(row["tools_json"]), json.loads(row["expected_json"]),
                                             truncated=not ids or ids[-1] not in eos_ids)
                        results.append({"id": row["id"], "category": row["category"], "response": text, **score})
            finally:
                if had_checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.train(was_training)
            group_scores = {}
            for name in ("call", "no_call"):
                group = [r for r in results if (r["category"] == "no_call") == (name == "no_call")]
                if group:
                    group_scores[name] = sum(r["exact"] for r in group)/len(group)
            metric = sum(group_scores.values()) / len(group_scores)
            record = {"step": state.global_step, "balanced_exact": metric, "groups": group_scores,
                      "elapsed_seconds": time.monotonic()-started}
            with (output / "validation_history.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            write_json(output / f"validation_step_{state.global_step}.json", results)
            if metric > self.best:
                self.best, self.best_step = metric, state.global_step
                model.save_pretrained(output / "best_adapter")
                tokenizer.save_pretrained(output / "best_adapter")
                write_json(output / "selection.json", {**record, "criterion": "ToolACE balanced strict accuracy",
                    "tie_break": "earliest step", "adapter": "best_adapter"})

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % cfg["validation_steps"] == 0:
                self.evaluate(state, model)

        def on_train_end(self, args, state, control, model=None, **kwargs):
            self.evaluate(state, model)

        def on_log(self, args, state, control, logs=None, **kwargs):
            with (output / "training_history.jsonl").open("a") as f:
                f.write(json.dumps({"step": state.global_step, **(logs or {})})+"\n")
            if state.global_step % 10 == 0:
                gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                                               "--format=csv,noheader,nounits"], text=True).strip()
                record = {"step": state.global_step, "gpu": gpu,
                          "disk_free_gib": shutil.disk_usage(ROOT).free / 2**30}
                with (output / "resources.jsonl").open("a") as f:
                    f.write(json.dumps(record)+"\n")

    callback = Validation()
    trainer = GRPOTrainer(model=model, processing_class=tokenizer,
        args=GRPOConfig(output_dir=str(output), **cfg["training"]),
        train_dataset=Dataset.from_list(train_rows), peft_config=lora,
        reward_funcs=make_reward(tokenizer, cfg["reward"], output, eos_ids), callbacks=[callback])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train()
    trainer.save_model(str(output / "last_adapter"))
    write_json(output / "summary.json", {"run": cfg["name"], "pilot": cfg.get("pilot", False),
        "training_seconds": time.monotonic()-started, "training_metrics": result.metrics,
        "steps": trainer.state.global_step, "selected_step": callback.best_step,
        "validation_balanced_exact": callback.best,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
        "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)})


if __name__ == "__main__":
    main()
