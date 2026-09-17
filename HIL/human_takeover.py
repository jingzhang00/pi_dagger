"""Minimal D1 human takeover controller.

The controller is deliberately independent from the RL implementation:
exo/glove input is converted to a safe absolute D1 command and can be used
from any inference loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Any

import numpy as np


DEFAULT_ARM_DELTA_LIMITS_RAD = (0.27, 0.38, 0.26, 0.08, 0.31, 0.08)


def _visual_notice(message: str, color: str = "\033[1;36m") -> None:
    print(f"{color}{message}\033[0m", flush=True)


@dataclass(frozen=True)
class HumanTakeoverConfig:
    exo_port: str
    glove_port: str
    control_hz: float = 10.0
    sync_arm_threshold_deg: float = 10.0
    sync_hand_threshold: float = 0.2
    sync_frames: int = 3
    deadband_arm_deg: float = 1.0
    deadband_hand: float = 0.02
    arm_delta_limits_rad: tuple[float, ...] = DEFAULT_ARM_DELTA_LIMITS_RAD
    max_abs_arm_rad: float = 3.14
    hand_min: float = 0.0
    hand_max: float = 1.0
    urdf_path: str | Path | None = None


class _TerminalKeys:
    """Non-blocking terminal key reader; safe to use when stdin is not a TTY."""

    def __init__(self) -> None:
        self.fd: int | None = None
        self._old: list[Any] | None = None

    def __enter__(self) -> "_TerminalKeys":
        if not sys.stdin.isatty():
            return self
        self.fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self) -> str | None:
        if self.fd is None or not select.select([self.fd], [], [], 0.0)[0]:
            return None
        return sys.stdin.read(1)

    def __exit__(self, *_: object) -> None:
        if self.fd is not None and self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)


def _state(name: str, value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (12,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 12-D vector, got {result.shape}")
    return result.copy()


def _safe_command(
    target: np.ndarray,
    current: np.ndarray,
    cfg: HumanTakeoverConfig,
) -> np.ndarray:
    target = _state("human target", target)
    current = _state("robot state", current)
    limits = np.asarray(cfg.arm_delta_limits_rad, dtype=np.float32)
    if limits.shape != (6,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("arm_delta_limits_rad must contain six positive finite values")
    arm = np.clip(target[:6], current[:6] - limits, current[:6] + limits)
    arm = np.clip(arm, -float(cfg.max_abs_arm_rad), float(cfg.max_abs_arm_rad))
    hand = np.clip(target[6:], float(cfg.hand_min), float(cfg.hand_max))
    return np.concatenate([arm, hand]).astype(np.float32)


class HumanTakeover:
    """Exoskeleton + glove takeover with guarded handoff synchronization."""

    def __init__(self, config: HumanTakeoverConfig) -> None:
        self.config = config
        self.exo: Any = None
        self.glove: Any = None
        self.available = False
        self.active = False
        self.syncing = False
        self._keys = _TerminalKeys()
        self._toggle_requested = False
        self._origin_exo: np.ndarray | None = None
        self._origin_glove: np.ndarray | None = None
        self._origin_robot: np.ndarray | None = None
        self._rebase_exo: np.ndarray | None = None
        self._rebase_glove: np.ndarray | None = None
        self._rebase_robot: np.ndarray | None = None
        self._hold_state: np.ndarray | None = None
        self._sync_count = 0
        self._last_exo_angles: np.ndarray | None = None
        self._sync_dashboard: Any = None

    def open(self) -> bool:
        """Open human devices. Missing ports disable HIL without stopping BH05."""
        ports = {"exo": self.config.exo_port, "glove": self.config.glove_port}
        missing = [
            f"{name}={port}"
            for name, port in ports.items()
            if not port or not Path(port).exists()
        ]
        if missing:
            _visual_notice(
                f"⚠ HIL unavailable; missing device port(s): {', '.join(missing)}",
                "\033[1;33m",
            )
            return False

        try:
            from beingbeyond_d1_edu_sdk.exo import ExoDriver, ExoDriverCfg
            from beingbeyond_d1_edu_sdk.glove_driver import GloveConfig, GloveReader

            self.exo = ExoDriver(
                ExoDriverCfg(port=self.config.exo_port, read_hz=self.config.control_hz)
            )
            self.exo.open()
            self.exo.release_torque()
            self.glove = GloveReader(
                GloveConfig(
                    port=self.config.glove_port,
                    timeout=0.01,
                    use_auto_calib=True,
                    offset_deg=[60, -70, 120, 120, 120, 120],
                )
            )
        except Exception:
            self.close()
            raise

        self._keys.__enter__()
        self.available = True
        print(
            "\n\033[1;36m[BH05]\033[0m  BASE VLA\n"
            "  CONTROL  \033[1;35m[h]\033[0m human takeover    "
            "\033[1;31m[q]\033[0m stop",
            flush=True,
        )
        return True

    def close(self) -> None:
        self.active = False
        self.syncing = False
        self._stop_sync_view()
        self._keys.__exit__(None, None, None)
        for obj in (self.glove, self.exo):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                logging.exception("Failed to close HIL device.")
        self.glove = self.exo = None
        self.available = False

    def poll_key(self) -> str | None:
        key = self._keys.read()
        if key == "h":
            self.request_toggle()
        return key

    def handle_key(self, key: str | int | None) -> None:
        if key == "h" or key == ord("h"):
            self.request_toggle()

    def request_toggle(self) -> None:
        self._toggle_requested = True

    def consume_toggle(self) -> bool:
        requested = self._toggle_requested
        self._toggle_requested = False
        return requested

    def begin_episode(self, robot_state: np.ndarray) -> None:
        """Record the human and robot origins before autonomous execution."""
        if not self.available:
            return
        self._origin_robot = _state("robot state", robot_state)
        self._origin_exo, self._origin_glove = self._read_human_pose()

    def activate(self, robot_state: np.ndarray) -> None:
        if not self.available:
            return
        if self._origin_robot is None:
            self.begin_episode(robot_state)
        self._hold_state = _state("robot state", robot_state)
        self._sync_count = 0
        self.syncing = True
        self.active = True
        _visual_notice("⏳⏳⏳  Human takeover selected; syncing  ⏳⏳⏳", "\033[1;35m")

    def start_sync_view(self) -> None:
        if self.active and self.syncing and self._sync_dashboard is None:
            self._start_sync_view()

    def deactivate(self) -> None:
        if self.active:
            _visual_notice("🤖  BH05 control restored.")
        self.active = False
        self.syncing = False
        self._stop_sync_view()
        self._hold_state = None
        self._sync_count = 0

    def toggle(self, robot_state: np.ndarray) -> None:
        if self.active:
            self.deactivate()
        else:
            self.activate(robot_state)

    def step(self, robot_state: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return a safe absolute command and whether synchronization is ongoing."""
        if not self.active:
            raise RuntimeError("HIL is not active")
        state = _state("robot state", robot_state)

        if self.syncing:
            angles, glove = self._read_human_pose()
            target = self._target_from_origin(angles, glove)
            arm_error_deg = np.rad2deg(target[:6] - state[:6])
            hand_error = target[6:] - state[6:]
            in_range = (
                np.all(np.abs(arm_error_deg) <= self.config.sync_arm_threshold_deg)
                and np.all(np.abs(hand_error) <= self.config.sync_hand_threshold)
            )
            self._sync_count = self._sync_count + 1 if in_range else 0
            self._update_sync_view(state, target, arm_error_deg, hand_error)
            if self._sync_count < self.config.sync_frames:
                assert self._hold_state is not None
                return _safe_command(self._hold_state, state, self.config), True
            self._rebase_exo = angles
            self._rebase_glove = glove
            self._rebase_robot = state.copy()
            self.syncing = False
            self._stop_sync_view()
            _visual_notice("✅✅✅  Human control activated  ✅✅✅", "\033[1;32m")
            return _safe_command(state, state, self.config), False

        angles, glove = self._read_human_pose()
        target = self._target_from_rebase(angles, glove)
        return _safe_command(target, state, self.config), False

    def _start_sync_view(self) -> None:
        try:
            from .sync_dashboard import SyncDashboard

            urdf_path = self.config.urdf_path
            if urdf_path is None:
                from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path

                urdf_path = get_default_urdf_path()
            self._sync_dashboard = SyncDashboard(urdf_path)
            self._sync_dashboard.start()
        except Exception as exc:
            self._sync_dashboard = None
            logging.warning("HIL sync dashboard unavailable; using terminal status: %s", exc)

    def _update_sync_view(
        self,
        state: np.ndarray,
        target: np.ndarray,
        arm_error_deg: np.ndarray,
        hand_error: np.ndarray,
    ) -> None:
        if self._sync_dashboard is None:
            return
        try:
            self._sync_dashboard.render(
                state,
                target,
                arm_error_deg,
                hand_error,
                self.config.sync_arm_threshold_deg,
                self.config.sync_hand_threshold,
                self._sync_count,
                self.config.sync_frames,
            )
        except Exception as exc:
            logging.warning("HIL sync dashboard stopped; using terminal status: %s", exc)
            self._stop_sync_view()

    def _stop_sync_view(self) -> None:
        dashboard, self._sync_dashboard = self._sync_dashboard, None
        if dashboard is not None:
            try:
                dashboard.stop()
            except Exception:
                logging.exception("Failed to close HIL sync dashboard.")

    def _read_human_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.exo is None or self.glove is None:
            raise RuntimeError("HIL devices are not open")
        frame = self.exo.read_frame()
        if hasattr(frame, "reads"):
            valid = [read.ok and read.angle_deg is not None for read in frame.reads]
            if sum(valid) < 4:
                raise RuntimeError("too few valid exoskeleton joints")
            angles = np.asarray(
                [
                    float(read.angle_deg)
                    if ok
                    else (self._last_exo_angles[index] if self._last_exo_angles is not None else np.nan)
                    for index, (read, ok) in enumerate(zip(frame.reads, valid))
                ],
                dtype=np.float32,
            )
        else:
            if not getattr(frame, "ok", False):
                raise RuntimeError("invalid exoskeleton frame")
            angles = np.asarray(frame.angles_deg, dtype=np.float32)
        glove = np.asarray(self.glove.get_norm_values(), dtype=np.float32)
        if (
            angles.shape != (6,)
            or glove.shape != (6,)
            or not np.isfinite(angles).all()
            or not np.isfinite(glove).all()
        ):
            raise RuntimeError("invalid human control pose")
        self._last_exo_angles = angles.copy()
        return angles, np.clip(glove, 0.0, 1.0)

    def _target_from_origin(self, angles: np.ndarray, glove: np.ndarray) -> np.ndarray:
        if self._origin_robot is None or self._origin_exo is None or self._origin_glove is None:
            raise RuntimeError("HIL episode origin is not initialized")
        return np.concatenate(
            [
                self._origin_robot[:6] + np.deg2rad(angles - self._origin_exo),
                self._origin_robot[6:] + glove - self._origin_glove,
            ]
        ).astype(np.float32)

    def _target_from_rebase(self, angles: np.ndarray, glove: np.ndarray) -> np.ndarray:
        if self._rebase_robot is None or self._rebase_exo is None or self._rebase_glove is None:
            raise RuntimeError("HIL rebase is not initialized")
        arm_delta = np.deg2rad(angles - self._rebase_exo)
        arm_deadband = np.deg2rad(self.config.deadband_arm_deg)
        arm_delta[np.abs(arm_delta) < arm_deadband] = 0.0
        hand_delta = glove - self._rebase_glove
        hand_delta[np.abs(hand_delta) < self.config.deadband_hand] = 0.0
        return np.concatenate(
            [self._rebase_robot[:6] + arm_delta, self._rebase_robot[6:] + hand_delta]
        ).astype(np.float32)


def _self_check() -> None:
    cfg = HumanTakeoverConfig("", "", arm_delta_limits_rad=(0.1,) * 6)
    command = _safe_command(
        np.asarray([1, 1, 1, 1, 1, 1, 2, -1, 0.5, 0.5, 0.5, 0.5]),
        np.zeros(12),
        cfg,
    )
    assert np.allclose(command[:6], 0.1)
    assert np.allclose(command[6:], [1, 0, 0.5, 0.5, 0.5, 0.5])


if __name__ == "__main__":
    _self_check()
    print("HIL self-check passed")
