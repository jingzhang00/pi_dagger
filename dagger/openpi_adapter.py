"""Thin π0.5 inference adapter for the D1 online loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dagger.openpi_config import policy_train_config
from dagger.settings import Config


class OpenPIAdapter:
    def __init__(self, settings: Config, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(f"π0.5 checkpoint has no params/: {checkpoint}")
        if not (checkpoint / "assets").is_dir():
            raise FileNotFoundError(f"π0.5 checkpoint has no assets/: {checkpoint}")
        from openpi.policies import policy_config

        self.settings = settings
        self.checkpoint = checkpoint
        self.action_dim = settings.action_dim
        config = policy_train_config(settings, checkpoint)
        self.policy = policy_config.create_trained_policy(
            config,
            checkpoint,
            default_prompt=str(settings.vla.get("prompt", "")),
            sample_kwargs={
                "num_steps": int(settings.vla.get("inference_steps", 10)),
            },
        )

    def infer(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if image.shape != (480, 640, 3):
            raise ValueError(f"policy image must be 480x640 RGB, got {image.shape}")
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError(f"policy state must be finite {self.action_dim}-D")
        result = self.policy.infer(
            {
                "observation/image": image,
                "observation/state": state,
                "prompt": str(self.settings.vla.get("prompt", "")),
            }
        )
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"policy returned an invalid action chunk: {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy returned non-finite actions")
        return actions

    def close(self) -> None:
        self.policy = None


def load_policy(settings: Config, checkpoint: str | Path) -> OpenPIAdapter:
    return OpenPIAdapter(settings, checkpoint)
