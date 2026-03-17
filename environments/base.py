from abc import ABC, abstractmethod
from typing import Optional


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

    @abstractmethod
    def run_llm_action(
        self,
        action: str,
        context_entries: list,
        output_dir: str,
        name: str,
        debug_dir: Optional[str] = None,
        parent_path: Optional[str] = None,
        scratchpad: str = "",
    ) -> Optional[str]:
        """Call the LLM to propose a new design and generate the design file.

        Args:
            action:          Action type string (e.g. 'gaussain', 'gaussian').
            context_entries: List of context dicts from build_context_entry().
            output_dir:      Directory where the design file will be written.
            name:            Design name (used as filename stem).
            debug_dir:       If set, write LLM prompts/responses here.
            parent_path:     Path to the parent design file (used by some envs
                             for modify-style actions).
            scratchpad:      Accumulated parameter-geometry knowledge string
                             maintained by the v2 framework. Empty string means
                             no scratchpad (default, used by all other frameworks).

        Returns:
            Path to the generated design file, or None on failure.
        """

    def set_llm_backend(self, backend, image_analyzer=None):
        """Swap the LLM backend (e.g. for Test-Time Discovery with Tinker).

        Override in environments whose agent supports pluggable backends.
        Default is a no-op that prints a warning.
        """
        print(f"Warning: {self.__class__.__name__} does not support set_llm_backend()")

    def get_results_csv_columns(self) -> list:
        """Return extra metric column names written to results.csv.

        The base columns (iteration, design, reward, best_reward, island) are
        always written; this method returns any additional env-specific columns.
        """
        return []

    def get_results_csv_row(self, metrics: dict) -> list:
        """Return extra metric values for a results.csv row.

        Values must match the order returned by get_results_csv_columns().
        """
        return []

    def get_reflection_inputs(self, design_path: str, case_dir: str) -> Optional[dict]:
        """Return inputs needed for the v2 reflection cycle, or None to skip.

        If non-None, must return a dict with keys:
            'intended_params'    -- dict loaded from LLM output (llm_params.json)
            'actual_action'      -- list of floats (the design vector actually used)
            'designer_analysis'  -- str from llm_analysis.txt
            'designer_reasoning' -- str from llm_rationale.txt
            'geometry_image_path'-- str path to geometry PNG, or None

        Default returns None (reflection skipped). Override in environments
        that support the v2 reflection cycle.
        """
        return None

    def sample_gaussian(self, mean_params: dict, output_dir: str, name: str,
                        std_scale: float = 1.0) -> Optional[str]:
        """Draw one Gaussian sample around mean_params and write to output_dir/name.json.

        Used by batch frameworks to generate N designs from a single LLM proposal.

        Args:
            mean_params: Parameter dict proposed by the LLM (before perturbation).
            output_dir:  Directory to write the sampled design file into.
            name:        Filename stem (without extension).
            std_scale:   Multiplicative scale factor applied to the environment's
                         base Gaussian std fractions.

        Returns:
            Path to the written design file.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sample_gaussian(). "
            "Override to enable batch Gaussian sampling.")

    def get_param_bounds(self) -> tuple:
        """Return (lb, ub) as numpy arrays for gradient-free optimizers (e.g. PSO).

        Returns:
            lb: np.ndarray of shape (n,) -- lower bounds per parameter
            ub: np.ndarray of shape (n,) -- upper bounds per parameter
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_param_bounds(). "
            "Override to enable gradient-free optimizers such as PSO.")

    def write_design(self, x, output_dir: str, name: str) -> str:
        """Write a design file from parameter vector x and return its path.

        Args:
            x:          1-D array-like of design parameters (matching get_param_bounds()).
            output_dir: Directory to write the design file into.
            name:       Filename stem (without extension).

        Returns:
            Path to the written design file.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement write_design(). "
            "Override to enable gradient-free optimizers such as PSO.")

    @staticmethod
    def add_args(parser):
        """Add environment-specific CLI arguments (optional override)."""
        pass
