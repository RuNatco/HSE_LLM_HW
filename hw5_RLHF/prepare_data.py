import argparse
import ast
import re
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from common import DATASET, MODEL, REVISION, ROOT, read_json, write_json, sha256
from common import canonical


def schema(value):
    if isinstance(value, list):
        return [schema(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {key: schema(item) for key, item in value.items() if item is not None}
    if isinstance(result.get("type"), str):
        result["type"] = {"dict": "object", "float": "number", "int": "integer",
                          "bool": "boolean", "tuple": "array", "any": "string"}.get(
                              result["type"], result["type"])
    return result


def extract_tools(system):
    marker = "Here is a list of functions in JSON format that you can invoke:"
    if marker not in system:
        raise ValueError("Unknown ToolACE system format")
    functions, _ = json.JSONDecoder().raw_decode(system.split(marker, 1)[1].lstrip())
    if not isinstance(functions, list) or not functions:
        raise ValueError("Missing tool list")
    names = [f["name"] for f in functions]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate function names")
    return [{"type": "function", "function": schema(f)} for f in functions]


def parse_reference(text, tools):
    text = text.strip()
    names = [t["function"]["name"] for t in tools]
    if not text.startswith("["):
        if any(re.search(re.escape(n) + r"\s*\(", text) for n in names) or "<tool" in text:
            raise ValueError("Unrecognized call-like reference")
        if not text:
            raise ValueError("Empty reference")
        return []
    aliases = {name: f"tool_{i}" for i, name in enumerate(names)}
    parts, pos = [], 0
    argument_aliases = {}
    while pos < len(text):
        if text[pos] in "\"'":
            quote, start = text[pos], pos
            pos += 1
            while pos < len(text):
                if text[pos] == "\\":
                    pos += 2
                elif text[pos] == quote:
                    pos += 1
                    break
                else:
                    pos += 1
            parts.append(text[start:pos])
            continue
        matched = False
        for name in sorted(names, key=len, reverse=True):
            if text.startswith(name, pos) and text[pos + len(name):].lstrip().startswith("("):
                parts.append(aliases[name])
                pos += len(name)
                matched = True
                break
        if not matched:
            keyword = re.match(r"([A-Za-z_]\w*)\s*(?==)", text[pos:])
            if keyword:
                name = keyword.group(1)
                alias = f"argument_{len(argument_aliases)}"
                argument_aliases[alias] = name
                parts.append(alias)
                pos += len(keyword.group(0))
                continue
            parts.append(text[pos])
            pos += 1
    node = ast.parse("".join(parts), mode="eval").body
    if not isinstance(node, ast.List) or not node.elts:
        raise ValueError("Expected nonempty list of calls")
    inverse = {v: k for k, v in aliases.items()}

    class JsonLiterals(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in {"true", "false", "null"}:
                return ast.copy_location(ast.Constant({"true": True, "false": False, "null": None}[node.id]), node)
            return node

    calls = []
    for call in node.elts:
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id not in inverse or call.args):
            raise ValueError("Unsupported function reference")
        arguments = {}
        for kw in call.keywords:
            original_name = argument_aliases.get(kw.arg, kw.arg)
            if kw.arg is None or original_name in arguments:
                raise ValueError("Duplicate/expanded argument")
            arguments[original_name] = ast.literal_eval(JsonLiterals().visit(kw.value))
        # Normalize tuples to JSON arrays, reject non-JSON literals.
        arguments = json.loads(json.dumps(arguments, allow_nan=False))
        calls.append({"name": inverse[call.func.id], "arguments": arguments})
    return calls



def describe(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "median": values[len(values)//2],
            "p95": values[int((len(values)-1)*0.95)], "max": values[-1],
            "mean": sum(values)/len(values)}


def expand(row, dialogue_id, tokenizer, limits):
    tools = extract_tools(row["system"])
    history, examples = [], []
    stats = Counter()
    lengths = {"prompt": [], "reference": []}
    for turn, message in enumerate(row["conversations"]):
        role, content = message["from"], message["value"]
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(f"Unknown role: {role}")
        if role != "assistant":
            history.append({"role": role, "content": content})
            continue
        calls = parse_reference(content, tools)
        answer = {"role": "assistant", "content": "" if calls else content}
        if calls:
            answer["tool_calls"] = [{"type": "function", "function": c} for c in calls]
        if history and history[-1]["role"] == "user":
            prompt = tokenizer.apply_chat_template(history, tools=tools, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            reference_length = len(tokenizer(content, add_special_tokens=False)["input_ids"])
            lengths["prompt"].append(len(ids))
            lengths["reference"].append(reference_length)
            category = "no_call" if not calls else "single" if len(calls) == 1 else "parallel"
            stats["candidate_" + category] += 1
            if len(ids) > limits["max_prompt_tokens"]:
                stats["filtered_long_prompt"] += 1
            else:
                examples.append({"id": f"{dialogue_id}:{turn}", "dialogue_id": dialogue_id,
                    "prompt": prompt, "tools_json": canonical(tools), "expected_json": canonical(calls),
                    "reference_text": content, "category": category, "prompt_tokens": len(ids)})
        else:
            stats["skipped_post_tool_or_non_user_answer"] += 1
        history.append(answer)
    return examples, stats, lengths, len(tools)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = read_json(args.config)["data"]
    out = ROOT / cfg["output_dir"]
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite prepared data: {out}")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    revision = HfApi().dataset_info(DATASET, revision=cfg.get("revision", "main")).sha
    ds = load_dataset(DATASET, revision=revision, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    unique, duplicates = {}, 0
    for row in ds:
        digest = hashlib.sha256(canonical(row).encode()).hexdigest()
        if digest in unique:
            duplicates += 1
        unique[digest] = row
    ids = sorted(unique)
    random.Random(cfg["seed"]).shuffle(ids)
    n_val = max(1, round(len(ids) * cfg["validation_fraction"]))
    split_ids = {"validation": ids[:n_val], "train": ids[n_val:]}
    out.mkdir(parents=True, exist_ok=True)
    report = {"source_dialogues": len(ds), "exact_duplicates_removed": duplicates, "splits": {}}
    rejected = []
    for split, dialogue_ids in split_ids.items():
        examples, stats = [], Counter()
        lengths = {"prompt": [], "reference": []}
        tool_counts = []
        for did in dialogue_ids:
            try:
                batch, counts, lens, nt = expand(unique[did], did, tokenizer, cfg)
            except (ValueError, SyntaxError, TypeError, KeyError, json.JSONDecodeError) as exc:
                rejected.append({"dialogue_id": did, "split": split, "reason": str(exc)})
                continue
            examples.extend(batch)
            stats.update(counts)
            tool_counts.append(nt)
            for key in lengths:
                lengths[key].extend(lens[key])
        random.Random(cfg["seed"]).shuffle(examples)
        report["splits"][split] = {"dialogues": len(dialogue_ids), "counts": dict(stats),
            "prompt_tokens_before_filter": describe(lengths["prompt"]),
            "reference_tokens_original_format": describe(lengths["reference"]),
            "tools_per_dialogue": describe(tool_counts)}
        write_json(out / f"{split}_dialogue_ids.json", dialogue_ids)
        if split == "validation":
            validation = examples
        else:
            train = examples
    seen = set()
    for split, examples in [("validation", validation), ("train", train)]:
        kept = []
        for example in examples:
            digest = hashlib.sha256(example["prompt"].encode()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                kept.append(example)
        report["splits"][split]["duplicate_prompts_removed"] = len(examples)-len(kept)
        if not kept:
            raise RuntimeError(f"No usable {split} examples")
        report["splits"][split]["retained_categories"] = dict(Counter(x["category"] for x in kept))
        (out / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in kept))
        write_json(out / f"{split}_examples.json", kept[:5])
    report["rejected_dialogues"] = len(rejected)
    write_json(out / "rejected_dialogues.json", rejected)
    write_json(out / "eda.json", report)
    write_json(out / "manifest.json", {"dataset": DATASET, "revision": revision,
        "model": MODEL, "model_revision": REVISION, "config": cfg,
        "files": {n: sha256(out/n) for n in ["train.jsonl", "validation.jsonl"]}})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
