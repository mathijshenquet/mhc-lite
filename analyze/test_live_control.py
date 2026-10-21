"""End-to-end check of live control on a tiny CPU run (shakespeare_char, GDSM): extend max_iters, change the learning
rate, save_now, refuse an unknown key and a bad value, then stop.  Exits non-zero on any mismatch."""

import json
import os
import re
import shutil
import subprocess
import sys
import time

OUT = "out-livetest-tiny-gdsm"
ARGS = [sys.executable, "-u", "train.py", "config/train_shakespeare_char.py", "--device=cpu", "--compile=False",
        "--dtype=float32", "--n_layer=2", "--n_head=2", "--n_embd=64", "--block_size=64", "--batch_size=4",
        "--gradient_accumulation_steps=1", "--max_iters=60", "--lr_decay_iters=60", "--warmup_iters=5",
        "--eval_interval=1000", "--eval_iters=2", "--log_interval=1", "--wandb_log=False",
        "--hyper_conn_type=gdsm", "--hyper_conn_n=4", "--out_prefix_dataset=livetest", "--out_prefix_model=tiny",
        "--out_prefix_method=gdsm", "--always_save_checkpoint=False"]


def write_control(payload):
    temporary = os.path.join(OUT, "control.json.tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle)
    os.replace(temporary, os.path.join(OUT, "control.json"))


def wait_ack(previous_iteration, timeout=120):
    path = os.path.join(OUT, "control.ack.json")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(path) as handle:
                ack = json.load(handle)
            if ack["iteration"] != previous_iteration:
                return ack
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise SystemExit("FAIL: no ack")


def last_iteration(log):
    found = re.findall(r"\[iter (\d+)\]", log)
    return int(found[-1]) if found else -1


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    log_path = os.path.join(OUT, "run.log")
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(ARGS, stdout=log_file, stderr=subprocess.STDOUT)
    while last_iteration(open(log_path).read()) < 10:
        time.sleep(0.2)
        if process.poll() is not None:
            raise SystemExit("FAIL: run ended early\n" + open(log_path).read()[-2000:])
    checks = []
    write_control({"max_iters": 100000, "learning_rate": 1e-3, "nonsense": 1, "grad_clip": -1})
    ack = wait_ack(None)
    checks.append(("max_iters applied", ack["applied"].get("max_iters", [None, None])[1] == 100000))
    checks.append(("learning_rate applied", ack["applied"].get("learning_rate", [None, None])[1] == 1e-3))
    checks.append(("unknown key refused", "nonsense" in ack["refused"]))
    checks.append(("negative grad_clip refused", "grad_clip" in ack["refused"]))
    first = ack["iteration"]
    write_control({"save_now": True})
    ack = wait_ack(first)
    checks.append(("save_now acknowledged", ack["actions"] == ["save_now"]))
    time.sleep(2)
    checks.append(("save_now wrote a snapshot", os.path.exists(os.path.join(OUT, f"ckpt-{ack['iteration']}.pt"))))
    while last_iteration(open(log_path).read()) < 80:     # beyond the original max_iters=60
        time.sleep(0.2)
        if process.poll() is not None:
            break
    checks.append(("ran past the original max_iters", last_iteration(open(log_path).read()) >= 80))
    second = ack["iteration"]
    write_control({"stop": True})
    wait_ack(second)
    code = process.wait(timeout=120)
    log = open(log_path).read()
    checks.append(("stop exits 0", code == 0))
    checks.append(("stop saved ckpt.pt", os.path.exists(os.path.join(OUT, "ckpt.pt"))))
    checks.append(("live lines in the log", log.count("[live ") == 3))
    for name, ok in checks:
        print(("ok   " if ok else "FAIL ") + name)
    shutil.rmtree(OUT, ignore_errors=True)
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
