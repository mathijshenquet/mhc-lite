"""Live run control through a JSON file next to the run's checkpoints.

Write `<out_dir>/control.json` (preferably to a temp file, then `mv`) with whitelisted settings and/or actions, e.g.
    {"max_iters": 2000, "ckpt_interval": 100}         or        {"save_now": true}        or        {"stop": true}
The run checks the file's mtime every iteration (one stat), applies what is allowed and writes
`<out_dir>/control.ack.json` with the iteration, what was applied (old → new), what was refused and why, and the
actions taken.  A control file that already exists when the run starts is not applied (it belongs to an earlier run).
"""

from __future__ import annotations

import json
import math
import os
import time

ACTIONS = ("save_now", "eval_now", "stop")


class LiveControl:
    def __init__(self, directory: str, settable: dict[str, type], log=print):
        self.path = os.path.join(directory, "control.json")
        self.ack_path = os.path.join(directory, "control.ack.json")
        self.settable = settable
        self.log = log
        try:
            self.seen_mtime = os.stat(self.path).st_mtime_ns
        except FileNotFoundError:
            self.seen_mtime = None

    def poll(self, namespace: dict, iteration: int) -> dict | None:
        """Apply a new control file to `namespace` (the training script's globals).  Returns None when nothing changed,
        else {"applied": {key: [old, new]}, "actions": {name: True}}."""
        try:
            mtime = os.stat(self.path).st_mtime_ns
        except FileNotFoundError:
            return None
        if mtime == self.seen_mtime:
            return None
        self.seen_mtime = mtime
        applied, refused, actions = {}, {}, {}
        try:
            with open(self.path) as handle:
                request = json.load(handle)
            if not isinstance(request, dict):
                raise ValueError("control.json must hold a JSON object")
        except (OSError, ValueError) as error:
            self._ack(iteration, applied, {"<file>": f"unreadable: {error}"}, actions)
            return None
        for key, value in request.items():
            if key in ACTIONS:
                if value is True:
                    actions[key] = True
                else:
                    refused[key] = "actions take the value true"
            elif key in self.settable:
                try:
                    new = self._coerce(key, value)
                except ValueError as error:
                    refused[key] = str(error)
                    continue
                applied[key] = [namespace[key], new]
                namespace[key] = new
            else:
                refused[key] = "not live-settable"
        self._ack(iteration, applied, refused, actions)
        return {"applied": applied, "actions": actions}

    def _coerce(self, key, value):
        wanted = self.settable[key]
        if wanted is bool:
            if not isinstance(value, bool):
                raise ValueError("expects true or false")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"expects a number ({wanted.__name__})")
        if wanted is int:
            if value != int(value):
                raise ValueError("expects an integer")
            value = int(value)
        else:
            value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("expects a finite, non-negative value")
        return value

    def _ack(self, iteration, applied, refused, actions):
        record = {"iteration": iteration, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "applied": applied, "refused": refused, "actions": list(actions)}
        temporary = self.ack_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(record, handle, indent=1)
        os.replace(temporary, self.ack_path)
        self.log(f"[live {iteration}] applied {applied} refused {refused} actions {list(actions)}")
