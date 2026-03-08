"""PegInsertionSide with larger clearance for GPU simulation compatibility.

GPU physics has slightly different precision than CPU, causing the tight 3mm
clearance to fail on GPU (~4% SR) vs CPU (~67% SR). This registers a variant
with configurable clearance.

Usage:
    import DPPO.peg_insertion_easy  # registers the env
    env = gym.make("PegInsertionSideEasy-v1", ...)
"""

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.registration import register_env


@register_env("PegInsertionSideEasy-v1", max_episode_steps=200, override=True)
class PegInsertionSideEasy(PegInsertionSideEnv):
    """PegInsertionSide with larger clearance (default 6mm vs original 3mm)."""
    _clearance = 0.006


@register_env("PegInsertionSideEasy4p5-v1", max_episode_steps=200, override=True)
class PegInsertionSideEasy4p5(PegInsertionSideEnv):
    """PegInsertionSide with 4.5mm radial clearance."""
    _clearance = 0.0045
