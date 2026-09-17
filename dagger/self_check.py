"""No-hardware smoke checks for the DAgger control invariants."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from dagger.dataset import sampler_weights
from dagger.online_runner import run_episode
from dagger.settings import load_config


class _Policy:
    def __init__(self):
        self.calls = 0

    def infer(self, image, state):
        self.calls += 1
        return np.tile(np.arange(12, dtype=np.float32), (4, 1))


class _Human:
    active = False

    def __init__(self):
        self.keys = iter([None, "h", None, None, "h", None, "q"])
        self.toggle = False
        self.syncing = True

    def begin_episode(self, state):
        pass

    def poll_key(self):
        key = next(self.keys, "q")
        if key == "h":
            self.toggle = True
        return key

    def consume_toggle(self):
        toggle, self.toggle = self.toggle, False
        return toggle

    def activate(self, state):
        self.active = True
        self.syncing = True

    def deactivate(self):
        self.active = False

    def step(self, state):
        syncing, self.syncing = self.syncing, False
        return state, syncing


class _Hardware:
    def reset(self, state):
        pass

    def observe(self):
        return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros(12, dtype=np.float32)

    def send(self, target, current=None):
        return np.asarray(target, dtype=np.float32)


class _Recorder:
    def __init__(self):
        self.frames = 0
        self.segments = 0

    def add(self, state, action, image, task):
        self.frames += 1

    def finish_segment(self, metadata=None):
        self.segments += 1
        return 1

    def discard_segment(self):
        pass


def main() -> None:
    weights = sampler_weights(100, 20, 0.5)
    assert np.isclose(weights[:100].sum(), 0.5)
    assert np.isclose(weights[100:].sum(), 0.5)

    settings = replace(load_config(), episode_seconds=1.0, control_hz=100.0)
    policy, human, hardware, recorder = _Policy(), _Human(), _Hardware(), _Recorder()
    result = run_episode(
        settings,
        policy,
        human,
        recorder,
        hardware,
        round_index=1,
        episode_index=0,
    )
    assert result["aborted"]
    assert recorder.frames >= 1, recorder.frames
    assert policy.calls >= 2, policy.calls
    assert recorder.segments == 1
    print("DAgger self-check passed")


if __name__ == "__main__":
    main()
