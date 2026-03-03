from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """Minimal interface that all simulation environments must implement."""

    @abstractmethod
    def simulate(self, design_path: str, case_dir: str, **kwargs) -> tuple:
        """Run simulation on a design file.

        Returns:
            (reward: float, results: dict) where results contains at minimum:
                'metrics'  -- dict of env-specific scalar values
                'images'   -- list of str paths to output images
                'feedback' -- str of qualitative analysis text
        """

    @abstractmethod
    def build_context_entry(self, db_entry) -> dict:
        """Convert a raw database row into a context dict for the LLM agent.

        Returns a dict with at minimum:
            'reward', 'ranking', 'images', 'feedback'
        plus env-specific keys (e.g. 'vector' for 2D, 'params' for 3D).
        """

    @abstractmethod
    def get_prompt_blocks(self) -> dict:
        """Return environment-specific prompt blocks.

        Must return a dict with:
            'format_context'              -- callable(context_list) -> str
            'format_response_instructions' -- callable(example_json) -> str
        plus any env-specific text the framework prompts may reference.
        """

    @staticmethod
    def add_args(parser):
        """Add environment-specific CLI arguments (optional override)."""
        pass
