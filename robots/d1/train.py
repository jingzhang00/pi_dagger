"""D1 clean π0.5 training entrypoint."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

from dagger.settings import load_config


OPENPI_ROOT = Path(__file__).resolve().parents[2] / "openpi"


def get_config(config_name: str | None = None, config_path: str | Path | None = None):
    if config_path is not None:
        os.environ["CONFIG"] = str(config_path)
    settings = load_config(config_path)
    from robots.d1.d1_config import get_d1_configs

    configs = {config.name: config for config in get_d1_configs()}
    name = config_name or str(settings.vla["config_name"])
    if name not in configs:
        raise ValueError(f"unknown D1 config {name!r}; available: {sorted(configs)}")
    return configs[name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", nargs="?")
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--fsdp-devices", type=int)
    args = parser.parse_args(argv)

    config = get_config(args.config_name, args.config_path)
    changes = {
        key: value
        for key, value in {
            "num_train_steps": args.steps,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "fsdp_devices": args.fsdp_devices,
        }.items()
        if value is not None
    }
    if changes:
        config = dataclasses.replace(config, **changes)

    from dagger.finetune_vla import run

    checkpoint = run(config, OPENPI_ROOT)
    print(f"CLEAN_CHECKPOINT={checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
