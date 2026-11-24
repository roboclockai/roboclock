"""Household configuration using the existing validated episode schema."""

from pathlib import Path

from roboclock.envs.lab.config import EpisodeSpec, sample_episode
from roboclock.envs.lab.config import load_config as validate_config

HOUSEHOLD_ROOT = Path(__file__).parent


def load_config(path=None):
    return validate_config(path or HOUSEHOLD_ROOT / "configs/benchmark.yaml")


__all__ = ["EpisodeSpec", "load_config", "sample_episode"]
