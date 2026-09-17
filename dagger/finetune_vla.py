"""Clean-policy OpenPI finetuning runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def latest_params_checkpoint(root: str | Path) -> Path:
    root = Path(root)
    candidates = [
        path
        for path in ([root] if (root / "params").is_dir() else [])
        + ([*root.iterdir()] if root.is_dir() else [])
        if path.is_dir() and (path / "params").is_dir()
    ]
    if not candidates:
        raise RuntimeError(f"training finished without a params checkpoint in {root}")
    return max(
        candidates,
        key=lambda path: (
            int(re.search(r"(?:step[_-])?(\d+)$", path.name).group(1))
            if re.search(r"(?:step[_-])?(\d+)$", path.name)
            else -1,
            path.name,
        ),
    )


def run(config, upstream_root: str | Path) -> Path:
    """Run OpenPI's trainer with a robot-provided clean-policy config."""
    trainer = _load(
        "dagger.openpi_training_train",
        Path(upstream_root) / "scripts" / "train.py",
    )
    trainer.main(config)
    return latest_params_checkpoint(config.checkpoint_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI for clean training; ``--round`` is a compatibility alias for one DAgger round."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--round" in arguments:
        from robots.d1.dagger_d1 import main as dagger_main

        return dagger_main(arguments)
    from robots.d1.train import main as train_main

    return train_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
