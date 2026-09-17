"""Small OpenPI config adapters shared by inference and DAgger training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dagger.settings import Config


@dataclass(frozen=True)
class DaggerDataConfig:
    """Runtime data config carrying the two datasets without copying either one."""

    data_config: Any
    clean_repo_id: str
    intervention_repo_id: str
    intervention_ratio: float

    def __getattr__(self, name: str) -> Any:
        return getattr(self.data_config, name)


@dataclass(frozen=True)
class DaggerDataFactory:
    base_factory: Any
    clean_repo_id: str
    intervention_repo_id: str
    intervention_ratio: float

    def create(self, assets_dirs, model_config) -> DaggerDataConfig:
        data_config = replace(
            self.base_factory.create(assets_dirs, model_config),
            repo_id=self.clean_repo_id,
        )
        return DaggerDataConfig(
            data_config=data_config,
            clean_repo_id=self.clean_repo_id,
            intervention_repo_id=self.intervention_repo_id,
            intervention_ratio=self.intervention_ratio,
        )


def clean_train_config(settings: Config):
    """Return the existing D1 π0.5 train config pointed at the clean dataset."""
    from robots.d1.d1_config import get_d1_configs

    config = get_d1_configs()[0]
    return replace(
        config,
        data=replace(config.data, repo_id=str(settings.clean_dataset)),
        checkpoint_base_dir=str(settings.clean_output),
    )


def policy_train_config(settings: Config, checkpoint: Path | str):
    """Build the data/model description needed to load a trained policy."""
    config = clean_train_config(settings)
    return replace(config, weight_loader=_no_op_weight_loader(), checkpoint_base_dir=str(Path(checkpoint).parent))


def dagger_train_config(settings: Config, init_checkpoint: Path | str, round_index: int):
    """Create one independent DAgger round initialized from ``init_checkpoint``."""
    from openpi.training import weight_loaders

    config = clean_train_config(settings)
    data = DaggerDataFactory(
        base_factory=config.data,
        clean_repo_id=str(settings.clean_dataset),
        intervention_repo_id=str(settings.intervention_dataset),
        intervention_ratio=settings.intervention_ratio,
    )
    finetune = settings.vla_finetune
    return replace(
        config,
        data=data,
        exp_name=f"round_{int(round_index):03d}",
        checkpoint_base_dir=str(settings.policy_output),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(Path(init_checkpoint) / "params")
        ),
        num_train_steps=int(finetune.get("steps", 5000)),
        batch_size=int(finetune.get("batch_size", 128)),
        save_interval=int(finetune.get("save_interval", 1000)),
        keep_period=finetune.get("keep_period", 15000),
        num_workers=int(finetune.get("num_workers", 8)),
        fsdp_devices=int(finetune.get("fsdp_devices", 1)),
        overwrite=True,
        resume=False,
    )


def _no_op_weight_loader():
    from openpi.training import weight_loaders

    return weight_loaders.NoOpWeightLoader()
