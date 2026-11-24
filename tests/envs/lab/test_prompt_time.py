"""XPOLICYLAB_PROMPT_TIME: every XPolicyLab client sends the training prompt of the roboclock-demo
fine-tunes, "<instruction> Time: <sim_time:.1f> s" (docs/five-domain-training.md, rule 4), and the
unchanged instruction otherwise."""

import numpy as np
import pytest

from roboclock.envs.lab.policies import g05, internw0, openwam, xpolicylab, xvla
from roboclock.envs.lab.policies.xpolicylab import policy_instruction, prompt_with_time

HOME = np.array([0, 0.5, 0.5, 0, 0, 0, 0.02, 0.02], dtype=np.float32)


def observation(sim_time):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    return {
        "rgb": image,
        "left_wrist_rgb": image,
        "right_wrist_rgb": image,
        "joint_positions": np.tile(HOME, 2),
        "instruction": "stack the bowls",
        "sim_time": np.array([sim_time], dtype=np.float32),
    }


def pack(module, monkeypatch, obs):
    attribute = "connection" if module is xpolicylab else "xpolicylab_connection"
    monkeypatch.setattr(module, attribute, lambda: object())
    policy = module.make()
    # State reset() would set from a server session; only the packing is under test here.
    policy.gripper = np.ones(2, dtype=np.float32)
    policy.vision = {}
    if module is xvla:
        return policy.observation(obs, b"")
    return policy.observation(obs)


def test_prompt_matches_the_training_format():
    # Training: demo frame k is sim time k / 25 s, formatted with one decimal.
    assert prompt_with_time("Do it.", 3037 / 25) == "Do it. Time: 121.5 s"
    assert policy_instruction(observation(7.0), True) == "stack the bowls Time: 7.0 s"
    assert policy_instruction(observation(7.0), False) == "stack the bowls"


@pytest.mark.parametrize("module", [xpolicylab, xvla, openwam, internw0, g05])
def test_every_client_appends_the_time_only_when_asked(module, monkeypatch):
    monkeypatch.delenv("XPOLICYLAB_PROMPT_TIME", raising=False)
    assert pack(module, monkeypatch, observation(12.44))["instruction"] == "stack the bowls"
    monkeypatch.setenv("XPOLICYLAB_PROMPT_TIME", "1")
    assert pack(module, monkeypatch, observation(12.44))["instruction"] == "stack the bowls Time: 12.4 s"
