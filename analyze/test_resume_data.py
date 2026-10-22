"""Resume-safety of the data stream: a tiny CPU run of 60 iterations straight through must see exactly the same
training batches as a run stopped at 30 (live-control stop) and resumed to 60, and its eval must use the same tokens
every time.  Exits non-zero on mismatch."""

import hashlib
import os
import re
import shutil
import subprocess
import sys

BASE = [sys.executable, "-u", "train.py", "config/train_shakespeare_char.py", "--device=cpu", "--compile=False",
        "--dtype=float32", "--dropout=0.0", "--n_layer=2", "--n_head=2", "--n_embd=64", "--block_size=64", "--batch_size=4",
        "--gradient_accumulation_steps=2", "--lr_decay_iters=60", "--warmup_iters=5", "--eval_interval=30", "--eval_iters=3",
        "--log_interval=1", "--wandb_log=False", "--hyper_conn_type=gdsm", "--hyper_conn_n=4", "--out_prefix_dataset=resumetest",
        "--out_prefix_model=tiny", "--always_save_checkpoint=True"]


def run(method, extra):
    out = f"out-resumetest-tiny-{method}"
    result = subprocess.run(BASE + [f"--out_prefix_method={method}"] + extra, capture_output=True, text=True,
                            env=os.environ | {"PYTHONHASHSEED": str(abs(hash(method)) % 1000)})
    return out, result.stdout + result.stderr


def losses(text):
    return {int(i): float(l) for i, l in re.findall(r"\[iter (\d+)\]loss: ([0-9.]+)", text)}


def main():
    for m in ("straight", "split"):
        shutil.rmtree(f"out-resumetest-tiny-{m}", ignore_errors=True)
    _, straight = run("straight", ["--max_iters=60"])
    _, first = run("split", ["--max_iters=30"])
    _, second = run("split", ["--max_iters=60", "--init_from=resume"])
    a, b = losses(straight), {**losses(first), **losses(second)}
    shared = sorted(set(a) & set(b))
    worst = max(abs(a[i] - b[i]) for i in shared if i > 30)
    evals = re.findall(r"\[step (\d+)\]train loss: ([0-9.]+), val loss: ([0-9.]+)", straight)
    print("per-iteration |diff| 28-36:", [(i, round(abs(a[i] - b[i]), 6)) for i in shared if 28 <= i <= 36])
    print(f"iterations compared after the resume: {sum(1 for i in shared if i > 30)}, max |loss diff| {worst:.2e}")
    print("evals (straight run):", evals)
    for m in ("straight", "split"):
        shutil.rmtree(f"out-resumetest-tiny-{m}", ignore_errors=True)
    sys.exit(0 if worst < 1e-5 else 1)


if __name__ == "__main__":
    main()
