from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin wrapper around train_task_modes.py that freezes the sequence backbone to precomputed PLM embeddings."
    )
    parser.add_argument("--plm-embedding-path", required=True, help="Torch file produced by cache_plm_embeddings.py")
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to train_task_modes.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    command = [
        sys.executable,
        str((root / "scripts" / "train_task_modes.py").resolve()),
        "--sequence-backbone",
        "precomputed_plm",
        "--plm-embedding-path",
        args.plm_embedding_path,
        *args.passthrough,
    ]
    completed = subprocess.run(command, cwd=root)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
