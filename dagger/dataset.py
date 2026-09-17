"""Intervention recording and clean/intervention sampling for DAgger."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from dagger.openpi_config import DaggerDataConfig
from dagger.settings import Config


DEFAULT_CAMERA = "observation.images.d1_rgb"
DEFAULT_STATE = "observation.state"
DEFAULT_ACTION = "action"
DEFAULT_DELTA = "action.delta"
DEFAULT_META_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def read_dataset_info(root: str | Path) -> dict[str, Any]:
    path = Path(root).expanduser() / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot dataset metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_d1_dataset(root: str | Path) -> dict[str, Any]:
    """Validate the schema at the trust boundary before opening hardware."""
    root = Path(root).expanduser()
    info = read_dataset_info(root)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{root} must be a LeRobot v2.1 dataset")
    if int(info.get("fps", 0)) != 10:
        raise ValueError(f"{root} must run at 10 Hz, got {info.get('fps')!r}")
    features = info.get("features", {})
    required = {
        DEFAULT_STATE: ("float32", [12]),
        DEFAULT_ACTION: ("float32", [12]),
        DEFAULT_DELTA: ("float32", [12]),
        DEFAULT_CAMERA: ("video", [480, 640, 3]),
    }
    for key, (dtype, shape) in required.items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise ValueError(f"{root} is missing feature {key!r}")
        if feature.get("dtype") != dtype or list(feature.get("shape", ())) != shape:
            raise ValueError(
                f"{root} feature {key!r} must be {dtype}{shape}, got {feature}"
            )
    if int(info.get("total_frames", 0)) < 1:
        raise ValueError(f"{root} contains no frames")
    return info


def sampler_weights(clean_size: int, intervention_size: int, ratio: float) -> np.ndarray:
    """Return per-item weights whose two dataset masses equal ``1-ratio``/``ratio``."""
    if clean_size < 1 or intervention_size < 1:
        raise ValueError("both datasets must contain at least one frame")
    if not 0.0 < ratio < 1.0:
        raise ValueError("intervention ratio must be strictly between zero and one")
    return np.concatenate(
        [
            np.full(clean_size, (1.0 - ratio) / clean_size, dtype=np.float64),
            np.full(intervention_size, ratio / intervention_size, dtype=np.float64),
        ]
    )


def _writer_args(root: Path, *, robot_type: str, fps: int) -> argparse.Namespace:
    append = (root / "meta" / "info.json").is_file()
    return argparse.Namespace(
        output_root=root,
        repo_id="d1_dagger_interventions",
        robot_type=robot_type,
        fps=fps,
        use_videos=True,
        append=append,
        image_writer_threads=0,
        video_codec="h264",
        video_pix_fmt="yuv420p",
        video_crf=23,
        video_gop=2,
        video_fast_decode=0,
        encoder_queue_maxsize=30,
        encoder_threads=None,
    )


class InterventionRecorder:
    """Append stable human-control segments to one persistent v2.1 dataset."""

    def __init__(self, settings: Config) -> None:
        from robots.collect_lerobot import LocalLeRobotV21Dataset

        info = validate_d1_dataset(settings.clean_dataset)
        features = {
            key: copy.deepcopy(value)
            for key, value in info["features"].items()
            if key not in DEFAULT_META_KEYS
        }
        self.root = settings.intervention_dataset
        self.writer = LocalLeRobotV21Dataset(
            args=_writer_args(
                self.root,
                robot_type=str(info.get("robot_type", "d1_edu")),
                fps=int(info["fps"]),
            ),
            features=features,
        )

    @property
    def total_frames(self) -> int:
        return int(self.writer.info.get("total_frames", 0))

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        image: np.ndarray,
        *,
        task: str,
    ) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        image = np.asarray(image, dtype=np.uint8)
        if state.shape != (12,) or action.shape != (12,):
            raise ValueError("intervention state and action must be 12-D")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("intervention state and action must be finite")
        if image.shape != (480, 640, 3):
            raise ValueError(f"intervention image must be 480x640 RGB, got {image.shape}")
        self.writer.add_frame(
            {
                DEFAULT_STATE: state,
                DEFAULT_ACTION: action,
                DEFAULT_DELTA: action - state,
                DEFAULT_CAMERA: image,
                "task": task,
            }
        )

    def finish_segment(self, metadata: dict[str, Any] | None = None) -> int:
        if not self.writer.has_pending_frames():
            return 0
        count = int(self.writer.episode_buffer["size"])
        self.writer.save_episode(metadata)
        return count

    def discard_segment(self) -> None:
        if self.writer.has_pending_frames():
            self.writer.clear_episode_buffer()

    def finalize(self) -> None:
        self.discard_segment()
        self.writer.finalize()


def install_mixed_loader() -> None:
    """Patch only this trainer process; clean-policy training stays untouched."""
    from openpi.training import data_loader

    if getattr(data_loader.create_torch_data_loader, "_dagger_mixed", False):
        return
    original = data_loader.create_torch_data_loader

    def create_torch_data_loader(data_config, *args, **kwargs):
        if isinstance(data_config, DaggerDataConfig):
            return create_mixed_data_loader(data_config, *args, **kwargs)
        return original(data_config, *args, **kwargs)

    create_torch_data_loader._dagger_mixed = True
    data_loader.create_torch_data_loader = create_torch_data_loader


def create_mixed_data_loader(
    data_config: DaggerDataConfig,
    model_config,
    action_horizon: int,
    batch_size: int,
    *,
    sharding=None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
):
    """Build one weighted sampler over both datasets, without duplicating clean data."""
    import jax
    import torch
    from openpi.training import data_loader

    clean_config = data_config.data_config
    intervention_config = dataclasses_replace(clean_config, repo_id=data_config.intervention_repo_id)
    clean = data_loader.create_torch_dataset(clean_config, action_horizon, model_config)
    intervention = data_loader.create_torch_dataset(
        intervention_config, action_horizon, model_config
    )
    weights = sampler_weights(len(clean), len(intervention), data_config.intervention_ratio)
    combined = torch.utils.data.ConcatDataset([clean, intervention])
    transformed = data_loader.transform_dataset(
        combined, clean_config, skip_norm_stats=skip_norm_stats
    )
    if framework == "pytorch" and torch.distributed.is_initialized():
        local_batch_size = batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = batch_size // jax.process_count() if framework != "pytorch" else batch_size
    if local_batch_size < 1:
        raise ValueError("batch size is smaller than the process count")
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=max(len(combined), local_batch_size),
        replacement=True,
    )
    loader = data_loader.TorchDataLoader(
        transformed,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=False,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )
    return data_loader.DataLoaderImpl(clean_config, loader)


def dataclasses_replace(value, **changes):
    # Kept local so importing this module does not import OpenPI/dataclasses internals.
    import dataclasses

    return dataclasses.replace(value, **changes)
