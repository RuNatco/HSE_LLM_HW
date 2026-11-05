"""Collect completed runs without loading models or recomputing metrics."""
import csv
import json
from pathlib import Path

from prepare_data import dump


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)
    fields = ["experiment", "method", "r", "alpha", "dropout", "target_modules",
              "trainable_parameters", "total_parameters", "training_seconds", "optimizer_steps",
              "peak_allocated_gib", "peak_reserved_gib", "sampled_device_peak_mib",
              "best_validation_loss", "rouge2", "rougeL", "test_examples", "evaluation_seconds"]
    rows, predictions = [], {}
    names = json.loads(Path("config.json").read_text())["experiments"]
    for name in names:
        path = Path("runs") / name
        if not (path / "summary.json").exists():
            continue
        s = json.loads((path / "summary.json").read_text())
        lora = s.get("lora") or {}
        row = {key: s.get(key, "") for key in fields}
        row.update(r=lora.get("r", ""), alpha=lora.get("lora_alpha", ""),
                   dropout=lora.get("lora_dropout", ""),
                   target_modules=",".join(lora.get("target_modules", [])),
                   training_seconds=s["training_seconds_including_validation_and_checkpoints"])
        rows.append(row)
        if name in ["base", "full_ft", "lora_baseline"]:
            predictions[name] = {r["id"]: r for r in
                                 map(json.loads, (path / "predictions.jsonl").read_text().splitlines())}
    with (out / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    selected = json.loads(Path("data/manual_eval_ids.json").read_text())
    test = {r["id"]: r for r in map(json.loads, Path("data/test.jsonl").read_text().splitlines())}
    manual = [{"id": id_, "article": test[id_]["article"], "reference": test[id_]["reference"],
               "outputs": {name: by_id[id_]["prediction"] for name, by_id in predictions.items()}}
              for id_ in selected]
    dump(out / "manual_comparison.json", manual)
    print(f"Collected {len(rows)} completed runs; manual examples: {len(manual)}")


if __name__ == "__main__":
    main()
