"""Execute named experiments sequentially, with persistent status and logs."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


root = Path(__file__).resolve().parent
os.chdir(root)
lock = (root / ".stage.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
names = sys.argv[1:]
assert names
config = json.loads((root / "config.json").read_text())
assert all(name in config["experiments"] for name in names)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="1", HF_HOME=str(root / "cache/huggingface"),
           TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
(root / "logs").mkdir(exist_ok=True)
completed = []


def status(state, **extra):
    data = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "experiments": names, "completed": completed, **extra}
    temporary = root / "stage_status.tmp"
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(root / "stage_status.json")
    print(json.dumps(data), flush=True)


try:
    for name in names:
        if shutil.disk_usage(root).free < 20 * 2**30:
            raise RuntimeError("Less than 20 GiB disk space remains")
        status("running", current=name)
        with (root / "logs" / f"{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, "-u", "run_experiment.py", name],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} exited with code {result.returncode}")
        assert (root / "runs" / name / "summary.json").exists()
        completed.append(name)
    status("completed")
except Exception as exc:
    status("failed", error=str(exc))
    raise
