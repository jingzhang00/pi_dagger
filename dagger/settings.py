"""Configuration for the D1 π0.5 DAgger runner."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config") / "config.yaml"


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def dataset_action_dim(dataset: str | Path) -> int:
    """Read the per-timestep action width from a local LeRobot dataset."""
    dataset = Path(dataset).expanduser()
    info_path = dataset / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        shape = info.get("features", {}).get("action", {}).get("shape")
        if isinstance(shape, list) and len(shape) == 1 and int(shape[0]) > 0:
            return int(shape[0])

    import pyarrow.parquet as parquet

    tables = sorted((dataset / "data").glob("chunk-*/episode_*.parquet"))
    if tables:
        value = parquet.read_table(tables[0], columns=["action"])["action"][0].as_py()
        if isinstance(value, list) and value:
            return len(value)
    raise FileNotFoundError(f"cannot determine action dimension from dataset {dataset}")


@dataclass(frozen=True)
class Config:
    path: Path
    robot_path: Path
    root: Path
    task: str
    project_name: str
    wandb_enabled: bool
    log_interval: int
    control_hz: float
    action_horizon: int
    chunk_steps: int
    episode_seconds: float
    initial_state_file: Path
    head_target_rad: tuple[float, float]
    vla: dict[str, Any]
    dagger: dict[str, Any]
    runtime: dict[str, Any]
    hardware: dict[str, Any]

    @property
    def action_dim(self) -> int:
        return dataset_action_dim(self.clean_dataset)

    @property
    def vla_finetune(self) -> dict[str, Any]:
        """Compatibility view used by the clean-policy training entrypoint."""
        return dict(self.vla.get("finetune", {}))

    @property
    def clean_dataset(self) -> Path:
        return _path(self.vla["dataset"], self.root)

    @property
    def clean_checkpoint(self) -> Path:
        configured = _path(self.vla["checkpoint"], self.root)
        if (configured / "params").is_dir() and (configured / "assets").is_dir():
            return configured
        candidates = [
            path
            for path in configured.iterdir()
            if path.is_dir() and (path / "params").is_dir() and (path / "assets").is_dir()
        ] if configured.is_dir() else []
        if candidates:
            return max(
                candidates,
                key=lambda path: (
                    int(re.search(r"\d+$", path.name).group())
                    if re.search(r"\d+$", path.name)
                    else -1,
                    path.name,
                ),
            )
        return configured

    @property
    def clean_output(self) -> Path:
        return _path(self.vla.get("output", "checkpoints/vla"), self.root)

    @property
    def base_checkpoint(self) -> Path:
        return _path(self.vla.get("base_checkpoint", ""), self.root)

    @property
    def intervention_dataset(self) -> Path:
        return _path(self.dagger["intervention_dataset"], self.root)

    @property
    def policy_output(self) -> Path:
        return _path(self.dagger["policy_output"], self.root)

    @property
    def state_file(self) -> Path:
        return _path(self.dagger["state_file"], self.root)

    @property
    def episodes_per_round(self) -> int:
        return int(self.dagger["episodes_per_round"])

    @property
    def intervention_ratio(self) -> float:
        return float(self.dagger["intervention_ratio"])

    @property
    def rounds(self) -> int:
        return int(self.dagger.get("rounds", 1))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path or os.environ.get("CONFIG") or DEFAULT_CONFIG).expanduser().resolve()
    data = _load_yaml(config_path)
    robot_path = Path(
        os.environ.get("ROBOT_CONFIG") or config_path.with_name("robot.yaml")
    ).expanduser().resolve()
    robot_data = _load_yaml(robot_path)
    # The checked-in file lives in <workspace>/dagger/config/, while tests and
    # deployments may pass a standalone config file.
    root = (
        config_path.parents[2]
        if config_path.parent.name == "config" and config_path.parent.parent.name == "dagger"
        else config_path.parent
    )

    vla = dict(data.get("vla", {}))
    dagger = dict(data.get("dagger", {}))
    runtime = {**dict(data.get("runtime", {})), **dict(robot_data.get("runtime", {}))}
    hardware = dict(robot_data.get("hardware", {}))

    control_hz = float(runtime.get("control_hz", 10))
    action_horizon = int(vla.get("action_horizon", 20))
    chunk_steps = int(runtime.get("chunk_steps", 2))
    episode_seconds = float(runtime.get("episode_seconds", 30))
    max_arm_delta_deg = float(runtime.get("max_arm_delta_deg", 12))
    sync_frames = int(runtime.get("human_sync_frames", 3))
    initial_state_file = _path(
        runtime.get(
            "initial_state_file",
            str(config_path.with_name("pick_up_the_doll_initial_state.json")),
        ),
        root,
    )
    head_target = tuple(
        float(value) for value in runtime.get("head_target_rad", (-0.2617994, -1.0471975))
    )

    if control_hz <= 0 or action_horizon < 1 or chunk_steps < 1:
        raise ValueError("control_hz, action_horizon and chunk_steps must be positive")
    if chunk_steps > action_horizon:
        raise ValueError(
            f"runtime.chunk_steps={chunk_steps} cannot exceed "
            f"vla.action_horizon={action_horizon}"
        )
    if (
        episode_seconds <= 0
        or max_arm_delta_deg <= 0
        or not math.isfinite(max_arm_delta_deg)
        or sync_frames < 1
        or len(head_target) != 2
        or not all(math.isfinite(v) for v in head_target)
    ):
        raise ValueError(
            "episode_seconds/max_arm_delta_deg must be positive, human_sync_frames "
            "must be positive, and head_target_rad must contain two finite values"
        )
    log_interval = int(data.get("log_interval", 100))
    if log_interval < 1:
        raise ValueError("log_interval must be positive")

    episodes = int(dagger.get("episodes_per_round", 5))
    rounds = int(dagger.get("rounds", 1))
    ratio = float(dagger.get("intervention_ratio", 0.5))
    if episodes < 1 or rounds < 1:
        raise ValueError("dagger rounds and episodes_per_round must be positive")
    if not 0.0 < ratio < 1.0:
        raise ValueError("dagger.intervention_ratio must be strictly between 0 and 1")
    if not vla.get("checkpoint"):
        raise ValueError("vla.checkpoint is required")
    if not vla.get("dataset"):
        raise ValueError("vla.dataset is required")
    for key in ("intervention_dataset", "policy_output", "state_file"):
        if not dagger.get(key):
            raise ValueError(f"dagger.{key} is required")

    return Config(
        path=config_path,
        robot_path=robot_path,
        root=root,
        task=str(data.get("task", "d1")),
        project_name=str(data.get("project_name", "openpi")),
        wandb_enabled=bool(data.get("wandb_enabled", True)),
        log_interval=log_interval,
        control_hz=control_hz,
        action_horizon=action_horizon,
        chunk_steps=chunk_steps,
        episode_seconds=episode_seconds,
        initial_state_file=initial_state_file,
        head_target_rad=head_target,
        vla=vla,
        dagger=dagger,
        runtime=runtime,
        hardware=hardware,
    )
