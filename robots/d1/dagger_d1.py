"""One D1 DAgger fine-tuning round."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dagger.settings import load_config


def latest_checkpoint(root: str | Path) -> Path | None:
    root = Path(root)
    candidates = []
    if (root / "params").is_dir():
        candidates.append(root)
    if root.is_dir():
        candidates.extend(
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "params").is_dir()
        )
    if not candidates:
        return None

    def key(path: Path) -> tuple[int, str]:
        match = re.search(r"(?:step[_-])?(\d+)$", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return max(candidates, key=key)


def train_round(settings, init_checkpoint: str | Path, round_index: int) -> Path:
    if int(round_index) < 1:
        raise ValueError("round must be positive")
    init_checkpoint = Path(init_checkpoint).expanduser().resolve()
    if not (init_checkpoint / "params").is_dir():
        raise FileNotFoundError(f"initial checkpoint has no params/: {init_checkpoint}")
    intervention = settings.intervention_dataset / "meta" / "info.json"
    if not intervention.is_file():
        raise FileNotFoundError(f"intervention dataset is missing: {intervention}")

    from dagger.dataset import install_mixed_loader, validate_d1_dataset
    from dagger.finetune_vla import run
    from dagger.openpi_config import dagger_train_config

    validate_d1_dataset(settings.clean_dataset)
    info = validate_d1_dataset(settings.intervention_dataset)
    if int(info.get("total_frames", 0)) < 1:
        raise ValueError("cannot fine-tune a DAgger round without intervention frames")
    install_mixed_loader()
    config = dagger_train_config(settings, init_checkpoint, round_index)
    output = run(config, Path(__file__).resolve().parents[2] / "openpi")
    if output is None:
        raise RuntimeError(f"OpenPI finished without a params checkpoint in {config.checkpoint_dir}")
    print(f"DAGGER_CHECKPOINT={output}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args(argv)
    if args.config:
        os.environ["CONFIG"] = str(args.config)
    settings = load_config(args.config)
    init_checkpoint = Path(args.init_checkpoint).expanduser() if args.init_checkpoint else None
    if init_checkpoint is None and settings.state_file.is_file():
        state = json.loads(settings.state_file.read_text(encoding="utf-8"))
        if state.get("checkpoint"):
            init_checkpoint = Path(state["checkpoint"])
    output = train_round(settings, init_checkpoint or settings.clean_checkpoint, args.round)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
