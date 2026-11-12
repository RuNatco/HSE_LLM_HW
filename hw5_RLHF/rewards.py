import json
from collections import Counter
from pathlib import Path

from common import canonical


def normalize_numbers(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [normalize_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_numbers(item) for key, item in value.items()}
    return value


def f1_counts(predicted, expected):
    denom = sum(predicted.values()) + sum(expected.values())
    return 2 * sum((predicted & expected).values()) / denom if denom else 1.0


def components(calls, expected, valid=True):
    if not valid:
        return {"exact": 0.0, "names": 0.0, "arguments": 0.0, "strict": 0.0, "shaped": 0.0}
    calls = normalize_numbers(calls)
    expected = normalize_numbers(expected)
    exact = float(Counter(map(canonical, calls)) == Counter(map(canonical, expected)))
    if not expected:
        return {"exact": exact, "names": exact, "arguments": exact, "strict": exact, "shaped": exact}
    names = f1_counts(Counter(c["name"] for c in calls), Counter(c["name"] for c in expected))
    argument_score = 0.0
    if calls:
        from scipy.optimize import linear_sum_assignment
        weights = []
        for predicted in calls:
            row = []
            for target in expected:
                if predicted["name"] != target["name"]:
                    row.append(0.0)
                else:
                    p = Counter((k, canonical(v)) for k, v in predicted["arguments"].items())
                    t = Counter((k, canonical(v)) for k, v in target["arguments"].items())
                    row.append(f1_counts(p, t))
            weights.append(row)
        rows, cols = linear_sum_assignment(weights, maximize=True)
        argument_score = 2 * sum(weights[i][j] for i, j in zip(rows, cols)) / (len(calls)+len(expected))
    return {"exact": exact, "names": names, "arguments": argument_score, "strict": exact,
            "shaped": 0.5*exact + 0.2*names + 0.3*argument_score}


class RewardEngine:
    def __init__(self, tokenizer):
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
        self.parser = Qwen3CoderToolParser(tokenizer)
        self.tokenizer = tokenizer

    def score(self, text, tools, expected, truncated=False):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        text = text.removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()
        valid, calls, error = False, [], None
        try:
            request = ChatCompletionRequest.model_validate({"model": "toolace-reward",
                "messages": [{"role": "user", "content": ""}], "tools": tools})
            self.parser.tools = request.tools
            response = self.parser.extract_tool_calls(text, request)
            calls = [{"name": c.function.name, "arguments": json.loads(c.function.arguments)}
                     for c in response.tool_calls]
            balanced = all(text.count(a) == text.count(b) for a, b in [
                ("<tool_call>", "</tool_call>"), ("<function=", "</function>"),
                ("<parameter=", "</parameter>")])
            valid = bool(text) and not truncated and balanced and text.count("<tool_call>") == len(calls)
            if not calls and any(tag in text for tag in ("<tool", "<function", "<parameter")):
                valid = False
            if any(not isinstance(c["arguments"], dict) for c in calls):
                valid = False
            canonical(calls)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            valid = False
            error = str(exc)
        return {**components(calls, expected, valid), "valid": valid, "calls": calls, "error": error}


def make_reward(tokenizer, kind, log_dir, eos_ids):
    engine = RewardEngine(tokenizer)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def toolace_reward(completions, completion_ids, expected_json, tools_json,
                       trainer_state=None, **kwargs):
        step = trainer_state.global_step if trainer_state else 0
        records = []
        for ids, target, tools in zip(completion_ids, expected_json, tools_json, strict=True):
            text = tokenizer.decode(ids, skip_special_tokens=False)
            result = engine.score(text, json.loads(tools), json.loads(target),
                                  truncated=not ids or ids[-1] not in eos_ids)
            records.append({"response": text, "expected": json.loads(target), **result})
        summary = {"step": step, "count": len(records), "kind": kind}
        for key in ["strict", "shaped", "exact", "names", "arguments", "valid"]:
            summary[key] = sum(r[key] for r in records)/len(records)
        with (log_dir / "reward_history.jsonl").open("a") as f:
            f.write(json.dumps(summary)+"\n")
        if step % 10 == 0:
            with (log_dir / "reward_examples.jsonl").open("a") as f:
                for record in records[:2]:
                    f.write(json.dumps({"step": step, **record}, ensure_ascii=False)+"\n")
        return [r[kind] for r in records]

    return toolace_reward


def render(calls):
    parts = []
    for call in calls:
        parameters = "\n".join(f"<parameter={key}>\n{value if isinstance(value, str) else json.dumps(value)}\n</parameter>"
                               for key, value in call["arguments"].items())
        parts.append(f"<tool_call>\n<function={call['name']}>\n{parameters}\n</function>\n</tool_call>")
    return "\n".join(parts)



def main():
    from common import MODEL, REVISION, read_jsonl, write_json
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    engine = RewardEngine(tokenizer)
    rows = read_jsonl("data/prepared/validation.jsonl")
    selected = []
    for category in ["single", "parallel", "no_call"]:
        selected.extend([r for r in rows if r["category"] == category][:2])
    results = []
    for row in selected:
        expected, tools = json.loads(row["expected_json"]), json.loads(row["tools_json"])
        correct = render(expected) if expected else row["reference_text"]
        variants = [("reference", correct, False), ("empty", "", False), ("truncated", correct, True)]
        if expected:
            variants.append(("missing_call", "Please provide more information.", False))
            wrong = json.loads(json.dumps(expected))
            wrong[0]["name"] = "nonexistent_function"
            variants.append(("wrong_function", render(wrong), False))
            wrong_args = json.loads(json.dumps(expected))
            wrong_args[0]["arguments"]["unexpected_argument"] = "incorrect"
            variants.append(("extra_argument", render(wrong_args), False))
        else:
            variants.append(("unnecessary_call", render([{"name": tools[0]["function"]["name"], "arguments": {}}]), False))
        for label, text, truncated in variants:
            score = engine.score(text, tools, expected, truncated)
            results.append({"id": row["id"], "category": row["category"], "variant": label,
                            "response": text, "expected": expected, **score})
            if label == "reference" and score["strict"] != 1:
                raise RuntimeError(f"Reference/parser round trip failed: {row['id']}; inspect tool schemas")
            if label in {"empty", "truncated", "missing_call", "unnecessary_call"} and score["strict"] != 0:
                raise RuntimeError(f"Invalid reward on {label}")
    write_json("data/reward_examples.json", results)
    print(f"Saved and checked {len(results)} reward examples")


if __name__ == "__main__":
    main()
