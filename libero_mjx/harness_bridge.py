"""Harness↔scripts bridge: emit structured results the harness tools can parse.

Scripts call `emit_result({...})` once at the end; the harness ComputeRunner
reads the `HARNESS_RESULT_JSON:` line from stdout. Also provides
`save_metrics(run_id, run_dir, payload)` to persist a metrics JSON the
`read_metrics` tool can retrieve.

This keeps the existing scripts unchanged in behavior — the marker line is
additive and human-readable.
"""

from __future__ import annotations

import json
import os
import sys
import time


def emit_result(payload: dict) -> None:
    """Print a single HARNESS_RESULT_JSON:{...} line to stdout.

    The harness ComputeRunner scans stdout for this marker and parses the
    JSON. Existing human-readable prints are unaffected.
    """
    line = "HARNESS_RESULT_JSON:" + json.dumps(payload, default=str)
    print(line, flush=True)


def save_metrics(run_id: str, run_dir: str, payload: dict) -> str:
    """Persist a metrics JSON to <run_dir>/<run_id>.json and return the path."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())