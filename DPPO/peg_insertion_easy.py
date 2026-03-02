"""PegInsertionSide with larger clearance for GPU simulation compatibility.

GPU physics has slightly different precision than CPU, causing the tight 3mm
clearance to fail on GPU (~4% SR) vs CPU (~67% SR). This registers a variant
with configurable clearance.

Usage:
    import DPPO.peg_insertion_easy  # registers the env
    env = gym.make("PegInsertionSideEasy-v1", ...)
"""

import gymnasium as gym
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv


class PegInsertionSideEasy(PegInsertionSideEnv):
    """PegInsertionSide with larger clearance (default 6mm vs original 3mm)."""
    _clearance = 0.006


# Register with gymnasium
gym.register(
    id="PegInsertionSideEasy-v1",
    entry_point="DPPO.peg_insertion_easy:PegInsertionSideEasy",
    max_episode_steps=200,
)
