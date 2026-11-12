import argparse
import shutil
from pathlib import Path
from common import MODEL, REVISION, ROOT, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run, output = Path(args.run), Path(args.output)
    config = read_json(run / "config.json")
    selection = read_json(run / "selection.json")
    if config["model"] != MODEL or config["revision"] != REVISION:
        raise SystemExit("Unexpected base model")
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Refusing to overwrite exported weights")
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise SystemExit("Need at least 20 GiB free before export")
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    base = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION,
        dtype=torch.bfloat16, device_map="cpu")
    adapted = PeftModel.from_pretrained(base, run / selection["adapter"])
    merged = adapted.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise RuntimeError("Unmerged LoRA parameters remain")
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(MODEL, revision=REVISION).save_pretrained(output)
    write_json(output / "export_manifest.json", {"base_model": MODEL, "base_revision": REVISION,
        "run": str(run), "selection": selection, "merged": True})
    print(f"Merged checkpoint: {output}")


if __name__ == "__main__":
    main()
