"""Checks that every queued job passes valid arguments to the training entry point.

In plain words: before switching the code under a queue of long GPU jobs, this reads
each job script in runs/queue/, extracts the options given to train_grpo.py, and
parses them with the current argument parser. A renamed or removed option is caught
here, in one second, instead of after hours of queue time. Run:
python -m src.tests.check_queue_jobs
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train.cli import build_parser  # noqa: E402


def job_args(script: str) -> list[list[str]]:
    """All argument lists passed to train_grpo.py in a job script (continuation lines joined)."""
    text = re.sub(r"\\\n\s*", " ", script)
    out = []
    for line in text.split("\n"):
        if "src/train/train_grpo.py" in line:
            tail = line.split("src/train/train_grpo.py", 1)[1].split(">>")[0].split(" > ")[0]
            tail = re.sub(r'"\$\w+"|\$\w+|"\$\{?\w+\}?[^"]*"', "X", tail)
            out.append(shlex.split(tail))
    return out


def main() -> None:
    parser = build_parser()
    jobs = sorted((ROOT / "runs" / "queue").glob("*.sh"))
    n = 0
    for job in jobs:
        for args in job_args(job.read_text()):
            parser.parse_args(args); n += 1
    print(f"[check_queue_jobs] OK — {n} commandes train_grpo.py de {len(jobs)} jobs en file parsées")


if __name__ == "__main__":
    main()
