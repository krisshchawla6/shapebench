from abc import ABC, abstractmethod
from typing import Callable


class BaseReward(ABC):
    """Abstract base for reward / fitness evaluation.

    Decouples reward computation — including how many simulations to run and
    with what parameters — from the environment's simulation primitive.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, run_sim: Callable, design_path: str, case_dir: str) -> tuple:
        """Evaluate fitness for a given design.

        Args:
            run_sim:      Environment's low-level simulation primitive.
                          Signature: run_sim(design_path, case_dir, **sim_kwargs) -> dict
                          The dict contains raw simulation outputs (env-specific).
            design_path:  Path to the design file.
            case_dir:     Directory for all outputs of this evaluation.

        Returns:
            (reward: float, results: dict) where results contains:
                'metrics'  -- dict of scalar values to log
                'images'   -- list of str paths to output images
                'feedback' -- str of qualitative analysis text
        """

    def get_prompt_blocks(self):
        """Return a prompt_blocks module/dict override, or None to use env defaults."""
        return None

    @staticmethod
    def add_args(parser):
        """Add reward-specific CLI arguments (optional override)."""
        pass
