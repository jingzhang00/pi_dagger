"""Online D1 π0.5 DAgger collection loop."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from dagger.actions import ARM_HIGH, ARM_LOW, safe_command
from dagger.dataset import InterventionRecorder, validate_d1_dataset
from dagger.openpi_adapter import OpenPIAdapter
from dagger.settings import Config, load_config

LOG = logging.getLogger(__name__)


def _load_initial_state(settings: Config) -> np.ndarray:
    data = json.loads(settings.initial_state_file.read_text(encoding="utf-8"))
    state = np.asarray(data.get("observation.state"), dtype=np.float32).reshape(-1)
    if state.shape != (12,) or not np.isfinite(state).all():
        raise ValueError("initial_state_file must contain a finite 12-D observation.state")
    return state


def _arm_bounds(settings: Config) -> tuple[np.ndarray, np.ndarray]:
    values = settings.runtime.get("arm_joint_limits_deg", {})
    if not isinstance(values, dict) or len(values) != 6:
        return ARM_LOW.copy(), ARM_HIGH.copy()
    try:
        limits = [values[f"joint_{index}"] for index in range(1, 7)]
        low = np.deg2rad([float(pair[0]) for pair in limits]).astype(np.float32)
        high = np.deg2rad([float(pair[1]) for pair in limits]).astype(np.float32)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("runtime.arm_joint_limits_deg must define six [low, high] pairs") from exc
    if np.any(low >= high) or not np.isfinite(np.concatenate([low, high])).all():
        raise ValueError("runtime.arm_joint_limits_deg must contain finite low < high pairs")
    return low, high


class D1Hardware:
    """Adapter exposing only the 12-D D1 state/action surface to the runner."""

    def __init__(self, settings: Config) -> None:
        from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path
        from robots.d1.d1_robot import D1Robot

        hardware = settings.hardware
        self.settings = settings
        self.low, self.high = _arm_bounds(settings)
        self.max_arm_delta_deg = float(settings.runtime.get("max_arm_delta_deg", 12.0))
        self.head_target = np.asarray(settings.head_target_rad, dtype=np.float32)
        self.robot = D1Robot(
            urdf_path=str(get_default_urdf_path()),
            arm_dev=str(hardware["arm_dev"]),
            arm_baud=int(hardware.get("arm_baud", 1_000_000)),
            hand_type=str(hardware.get("hand_type", "right")),
            hand_can=str(hardware.get("hand_can", "can0")),
            hand_baud=int(hardware.get("hand_baud", 1_000_000)),
            vision_device=str(hardware["camera_device"]),
        )
        try:
            self.robot.vision.view = "left"
            self.robot.vision.start()
        except Exception:
            self.robot.close()
            raise

    def _read_state(self) -> np.ndarray:
        state = np.asarray(self.robot.get_q(), dtype=np.float32).reshape(-1)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise RuntimeError(f"D1 returned an invalid 14-D joint state: {state.shape}")
        return state[2:].copy()

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(self.robot.vision.read_view(), dtype=np.uint8)
        if image.shape != (480, 640, 3):
            raise RuntimeError(f"D1 camera returned {image.shape}, expected (480, 640, 3)")
        # StereoRGBCamera returns OpenCV BGR; OpenPI and the dataset use RGB.
        return np.ascontiguousarray(image[:, :, ::-1]), self._read_state()

    def send(self, target: np.ndarray, current: np.ndarray | None = None) -> np.ndarray:
        current = self._read_state() if current is None else np.asarray(current, dtype=np.float32)
        command = safe_command(
            target,
            current,
            max_arm_delta_deg=self.max_arm_delta_deg,
            arm_low=self.low,
            arm_high=self.high,
        )
        self.robot.head_arm.set_positions(
            np.concatenate([self.head_target, command[:6]]).astype(np.float32)
        )
        self.robot.hand.set_joint_pos(command[6:].astype(np.float32))
        return command

    def reset(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (12,) or not np.isfinite(state).all():
            raise ValueError("reset state must be a finite 12-D vector")
        state = safe_command(
            state,
            state,
            max_arm_delta_deg=self.max_arm_delta_deg,
            arm_low=self.low,
            arm_high=self.high,
        )
        self.robot.head_arm.set_positions(
            np.concatenate([self.head_target, state[:6]]).astype(np.float32)
        )
        self.robot.hand.set_joint_pos(state[6:].astype(np.float32))

    def close(self) -> None:
        try:
            self.robot.vision.stop()
        finally:
            self.robot.close()


def _make_hil(settings: Config):
    from HIL.human_takeover import HumanTakeover, HumanTakeoverConfig

    runtime, hardware = settings.runtime, settings.hardware
    return HumanTakeover(
        HumanTakeoverConfig(
            exo_port=str(hardware.get("exo_port", "")),
            glove_port=str(hardware.get("glove_port", "")),
            control_hz=settings.control_hz,
            sync_arm_threshold_deg=float(runtime.get("human_sync_arm_threshold_deg", 10)),
            sync_hand_threshold=float(runtime.get("human_sync_hand_threshold", 0.2)),
            sync_frames=int(runtime.get("human_sync_frames", 3)),
            deadband_arm_deg=float(runtime.get("human_deadband_arm_deg", 1)),
            deadband_hand=float(runtime.get("human_deadband_hand", 0.02)),
        )
    )


def run_episode(
    settings: Config,
    policy: Any,
    human: Any,
    recorder: Any,
    hardware: Any,
    *,
    round_index: int,
    episode_index: int,
) -> dict[str, Any]:
    """Run one episode; fake objects can exercise this state machine without hardware."""
    initial = _load_initial_state(settings)
    hardware.reset(initial)
    image, state = hardware.observe()
    human.begin_episode(state)
    actions: np.ndarray | None = None
    action_offset = 0
    stable_frames = 0
    segments = 0
    aborted = False
    released = False
    policy_blocked = False
    deadline = time.monotonic() + settings.episode_seconds
    period = 1.0 / settings.control_hz

    while time.monotonic() < deadline:
        tick = time.monotonic()
        key = human.poll_key()
        if key == "q" or key == ord("q"):
            aborted = True
            break

        image, state = hardware.observe()
        if human.consume_toggle():
            actions = None
            action_offset = 0
            if human.active:
                human.deactivate()
                segments += recorder.finish_segment(
                    {
                        "round": round_index,
                        "episode": episode_index,
                        "source": "human_takeover",
                    }
                ) > 0
                released = True
            else:
                human.activate(state)
                start_sync_view = getattr(human, "start_sync_view", None)
                if start_sync_view is not None:
                    start_sync_view()

        if released:
            # Release is a hard cache barrier. The following tick gets the fresh observation.
            hardware.send(state, state)
            released = False
        elif human.active:
            try:
                command, syncing = human.step(state)
                sent = hardware.send(command, state)
                if not syncing:
                    recorder.add(state, sent, image, task=str(settings.vla["prompt"]))
                    stable_frames += 1
            except Exception:
                LOG.exception("HIL command failed; holding the current safe pose")
                hardware.send(state, state)
                human.deactivate()
                segments += recorder.finish_segment(
                    {
                        "round": round_index,
                        "episode": episode_index,
                        "source": "human_takeover",
                    }
                ) > 0
        else:
            if policy_blocked:
                hardware.send(state, state)
            elif actions is None or action_offset >= settings.chunk_steps:
                try:
                    actions = np.asarray(policy.infer(image, state), dtype=np.float32)
                    if (
                        actions.ndim != 2
                        or actions.shape[1] != settings.action_dim
                        or actions.shape[0] < settings.chunk_steps
                        or not np.isfinite(actions).all()
                    ):
                        raise ValueError(f"invalid policy action chunk {actions.shape}")
                    action_offset = 0
                except Exception:
                    LOG.exception("π0.5 inference failed; holding the current safe pose")
                    actions = None
                    action_offset = 0
                    policy_blocked = True
                    hardware.send(state, state)
                    elapsed = time.monotonic() - tick
                    time.sleep(max(0.0, period - elapsed))
                    continue
            if not policy_blocked:
                hardware.send(actions[action_offset], state)
                action_offset += 1

        elapsed = time.monotonic() - tick
        time.sleep(max(0.0, period - elapsed))

    if human.active:
        human.deactivate()
        segments += recorder.finish_segment(
            {
                "round": round_index,
                "episode": episode_index,
                "source": "human_takeover",
            }
        ) > 0
    else:
        recorder.discard_segment()
    return {
        "round": round_index,
        "episode": episode_index,
        "aborted": aborted,
        "stable_frames": stable_frames,
        "segments": int(segments),
    }


def _checkpoint_is_valid(path: Path) -> bool:
    return (path / "params").is_dir() and (path / "assets").is_dir()


def _resume_state(settings: Config) -> tuple[Path, int]:
    if settings.state_file.is_file():
        data = json.loads(settings.state_file.read_text(encoding="utf-8"))
        value = data.get("checkpoint")
        if value:
            checkpoint = Path(value)
            if not checkpoint.is_absolute():
                checkpoint = settings.root / checkpoint
            checkpoint = checkpoint.resolve()
            if _checkpoint_is_valid(checkpoint):
                return checkpoint, int(data.get("round", 0))
    checkpoint = settings.clean_checkpoint
    if not _checkpoint_is_valid(checkpoint):
        raise FileNotFoundError(
            f"clean π0.5 checkpoint must contain params/ and assets/: {checkpoint}"
        )
    return checkpoint, 0


def _write_state(settings: Config, checkpoint: Path, round_index: int) -> None:
    settings.state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.state_file.with_suffix(settings.state_file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "round": round_index,
                "checkpoint": str(checkpoint.resolve()),
                "intervention_dataset": str(settings.intervention_dataset.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(settings.state_file)


def validate_before_motion(settings: Config) -> None:
    validate_d1_dataset(settings.clean_dataset)
    if not settings.initial_state_file.is_file():
        raise FileNotFoundError(f"initial state file is missing: {settings.initial_state_file}")
    _load_initial_state(settings)
    _resume_state(settings)


def _run_finetune_subprocess(settings: Config, round_index: int, checkpoint: Path) -> Path:
    command = [
        sys.executable,
        "-m",
        "dagger.finetune_vla",
        "--config",
        str(settings.path),
        "--round",
        str(round_index),
        "--init-checkpoint",
        str(checkpoint),
    ]
    environment = os.environ.copy()
    environment["CONFIG"] = str(settings.path)
    paths = [str(settings.root), str(settings.root / "openpi" / "src")]
    environment["PYTHONPATH"] = os.pathsep.join(paths + [environment.get("PYTHONPATH", "")])
    subprocess.run(command, cwd=settings.root, env=environment, check=True)
    from robots.d1.dagger_d1 import latest_checkpoint

    output = latest_checkpoint(settings.policy_output / settings.vla["config_name"] / f"round_{round_index:03d}")
    if output is None:
        raise RuntimeError("DAgger subprocess completed without a params checkpoint")
    return output


def run(
    settings: Config,
    *,
    rounds: int | None = None,
    episodes_per_round: int | None = None,
) -> int:
    os.environ["CONFIG"] = str(settings.path)
    validate_before_motion(settings)
    checkpoint, completed_round = _resume_state(settings)
    round_count = int(rounds if rounds is not None else settings.rounds)
    episode_count = int(
        episodes_per_round if episodes_per_round is not None else settings.episodes_per_round
    )
    if round_count < 1 or episode_count < 1:
        raise ValueError("rounds and episodes_per_round must be positive")

    for offset in range(round_count):
        round_index = completed_round + offset + 1
        before_frames = 0
        if (settings.intervention_dataset / "meta" / "info.json").is_file():
            from dagger.dataset import read_dataset_info

            before_frames = int(read_dataset_info(settings.intervention_dataset).get("total_frames", 0))
        policy = human = hardware = recorder = None
        stop_requested = False
        try:
            policy = OpenPIAdapter(settings, checkpoint)
            human = _make_hil(settings)
            if not human.open():
                raise RuntimeError("HIL devices are unavailable; refusing to move the robot")
            recorder = InterventionRecorder(settings)
            hardware = D1Hardware(settings)
            for episode_index in range(episode_count):
                result = run_episode(
                    settings,
                    policy,
                    human,
                    recorder,
                    hardware,
                    round_index=round_index,
                    episode_index=episode_index,
                )
                LOG.info("episode result: %s", result)
                if result["aborted"]:
                    stop_requested = True
                    break
        finally:
            if recorder is not None:
                recorder.finalize()
            if human is not None:
                human.close()
            if hardware is not None:
                hardware.close()
            if policy is not None:
                policy.close()

        if stop_requested:
            _write_state(settings, checkpoint, completed_round)
            return 0

        after_frames = before_frames
        if (settings.intervention_dataset / "meta" / "info.json").is_file():
            from dagger.dataset import read_dataset_info

            after_frames = int(read_dataset_info(settings.intervention_dataset).get("total_frames", 0))
        if after_frames > before_frames:
            checkpoint = _run_finetune_subprocess(settings, round_index, checkpoint)
        _write_state(settings, checkpoint, round_index)
        completed_round = round_index
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--episodes-per-round", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    settings = load_config(args.config)
    os.environ["CONFIG"] = str(settings.path)
    validate_before_motion(settings)
    if args.dry_run:
        print("DAgger configuration and clean dataset/checkpoint validation passed")
        return 0
    return run(
        settings,
        rounds=args.rounds,
        episodes_per_round=args.episodes_per_round,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
