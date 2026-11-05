"""Freeze model/data revisions and identical, untruncated splits for all runs."""
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def messages(config, article):
    return [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": config["user_template"].format(article=article)},
    ]


def digest(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def main():
    config = json.loads(Path("config.json").read_text())
    out = Path("data")
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        raise RuntimeError("Prepared data already exists; preserve the frozen splits.")
    lock_path = Path("revisions.json")
    if lock_path.exists():
        revisions = json.loads(lock_path.read_text())
    else:
        api = HfApi()
        revisions = {
            "model": api.model_info(config["model"]).sha,
            "dataset": api.dataset_info(config["dataset"]).sha,
        }
        dump(lock_path, revisions)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revisions["model"])
    dataset = load_dataset(config["dataset"], config["dataset_config"], revision=revisions["dataset"])
    seen_articles, seen_ids = set(), set()
    stats, examples, selected = {}, {}, {}
    # Reserve evaluation texts before selecting training examples.
    for split in ["test", "validation", "train"]:
        want = config["split_sizes"][split]
        rows, counts = [], {"empty": 0, "duplicate": 0, "too_long": 0, "scanned": 0}
        for row in dataset[split].shuffle(seed=config["seed"]):
            counts["scanned"] += 1
            article, target = row["article"].strip(), row["highlights"].strip()
            key = digest(article)
            if not article or not target:
                counts["empty"] += 1
                continue
            if key in seen_articles or row["id"] in seen_ids:
                counts["duplicate"] += 1
                continue
            prompt = tokenizer.apply_chat_template(messages(config, article), tokenize=False, add_generation_prompt=True)
            answer = target + tokenizer.eos_token
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            if (len(prompt_ids) > config["max_prompt_tokens"] or
                    len(answer_ids) > config["max_target_tokens"] or
                    len(prompt_ids) + len(answer_ids) > config["max_length"]):
                counts["too_long"] += 1
                continue
            rows.append({"id": row["id"], "article": article, "reference": target,
                         "prompt": prompt, "input_ids": prompt_ids + answer_ids,
                         "labels": [-100] * len(prompt_ids) + answer_ids,
                         "prompt_tokens": len(prompt_ids), "target_tokens": len(answer_ids),
                         "article_sha256": key})
            seen_articles.add(key)
            seen_ids.add(row["id"])
            if len(rows) == want:
                break
        if len(rows) != want:
            raise RuntimeError(f"Only {len(rows)} eligible rows in {split}; need {want}")
        path = out / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = {
            **counts, "selected": len(rows), "source_size": len(dataset[split]),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / len(rows),
            "mean_target_tokens": sum(r["target_tokens"] for r in rows) / len(rows),
            "max_total_tokens": max(len(r["input_ids"]) for r in rows),
        }
        selected[split] = [r["id"] for r in rows]
        examples[split] = [{k: r[k] for k in ["id", "article", "reference"]} for r in rows[:3]]
        print(split, stats[split], flush=True)
    dump(out / "selected_ids.json", selected)
    dump(out / "examples.json", examples)
    dump(out / "manual_eval_ids.json", selected["test"][:10])
    dump(out / "manifest.json", {"config": config, "revisions": revisions, "splits": stats,
                                 "manual_selection": "First 10 seeded test examples, chosen before generation",
                                 "deduplication": "Exact article text after whitespace normalization and source IDs; no semantic deduplication"})
    print("DATA PREPARATION COMPLETED", flush=True)


if __name__ == "__main__":
    main()
