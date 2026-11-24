"""roboclock.envs.observation: images render once, when first read, at the observation's instant."""

import copy
import json
import pickle

import gymnasium as gym
import numpy as np
import pytest

from roboclock.envs.observation import Observation, StaleObservationError, image_keys


class Scene:
    def __init__(self):
        self.time, self.renders = 0.0, []

    def render(self):
        self.renders.append(self.time)
        return {"rgb": np.full((2, 3, 3), len(self.renders), np.uint8), "depth": np.zeros((2, 3))}


def observation(scene):
    state = {"joint_positions": np.zeros(16, np.float32), "instruction": "go"}
    return Observation(state, ("rgb", "depth"), scene.render, lambda: scene.time)


def test_state_reads_do_not_render():
    scene = Scene()
    obs = observation(scene)
    assert obs["joint_positions"].shape == (16,) and obs.get("instruction") == "go"
    assert "rgb" in obs and "missing" not in obs and len(obs) == 4
    assert set(obs) == set(obs.keys()) == {"joint_positions", "instruction", "rgb", "depth"}
    assert not obs.rendered and scene.renders == [] and "unrendered" in repr(obs)


def test_first_image_read_renders_once_at_the_observation_instant():
    scene = Scene()
    obs = observation(scene)
    assert obs["rgb"][0, 0, 0] == 1
    obs["depth"], obs.get("rgb"), obs["rgb"]
    assert scene.renders == [0.0] and obs.rendered and obs.render_seconds >= 0
    assert list(obs)[:2] == ["rgb", "depth"]  # images first, as eager observations


def test_reading_images_after_the_simulation_advanced_raises():
    scene = Scene()
    obs = observation(scene)
    scene.time = 0.02
    with pytest.raises(StaleObservationError, match="t=0.0000 s read at t=0.0200 s"):
        obs["rgb"]
    assert scene.renders == []
    assert obs["joint_positions"].shape == (16,)  # state stays readable


@pytest.mark.parametrize(
    "convert",
    [
        dict,
        lambda o: {**o},
        lambda o: o.copy(),
        lambda o: dict(o.items()),
        copy.copy,
        copy.deepcopy,
        lambda o: pickle.loads(pickle.dumps(o)),
    ],
)
def test_conversions_carry_every_key(convert):
    scene = Scene()
    plain = convert(observation(scene))
    assert type(plain) is dict and set(plain) == {"joint_positions", "instruction", "rgb", "depth"}
    assert scene.renders == [0.0]


def test_json_and_gym_space_see_every_key():
    scene = Scene()
    obs = observation(scene)
    space = gym.spaces.Dict(
        {
            "joint_positions": gym.spaces.Box(-np.inf, np.inf, (16,), np.float32),
            "instruction": gym.spaces.Text(8),
            "rgb": gym.spaces.Box(0, 255, (2, 3, 3), np.uint8),
            "depth": gym.spaces.Box(0, np.inf, (2, 3), np.float64),
        }
    )
    assert space.contains(obs)
    assert set(image_keys(space)) == {"rgb", "depth"}
    encoded = json.loads(json.dumps(observation(Scene()), default=lambda a: a.tolist()))
    assert set(encoded) == {"joint_positions", "instruction", "rgb", "depth"}


def test_writing_state_keeps_images_lazy_and_image_writes_render_first():
    scene = Scene()
    obs = observation(scene)
    obs["extra"] = 1
    obs.pop("extra")
    assert scene.renders == []
    obs["rgb"] = None
    assert scene.renders == [0.0] and obs["rgb"] is None and obs["depth"].shape == (2, 3)


def test_renderer_must_return_exactly_the_image_keys():
    obs = Observation({"a": 1}, ("rgb",), lambda: {"rgb": 0, "extra": 1}, lambda: 0.0)
    with pytest.raises(RuntimeError, match="Renderer returned"):
        obs["rgb"]
    with pytest.raises(ValueError, match="also state entries"):
        Observation({"rgb": 1}, ("rgb",), dict, lambda: 0.0)


def test_equality_compares_rendered_contents():
    def make(value):
        return Observation({"a": 1}, ("rgb",), lambda: {"rgb": value}, lambda: 0.0)

    assert make(5) == {"a": 1, "rgb": 5} and make(5) == make(5)
    assert make(5) != make(6) and make(5) != {"a": 1}
