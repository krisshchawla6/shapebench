# Strategy prompts for island_2d_gaussian framework.
# Environment-specific prompt blocks (format_context, format_response_instructions)
# are provided at runtime via env.get_prompt_blocks().

from .generate import GENERATE_SYSTEM, GENERATE_USER
from .generate_direct import GENERATE_DIRECT_SYSTEM, GENERATE_DIRECT_USER
from .modify import MODIFY_SYSTEM, MODIFY_USER
from .modify_direct import MODIFY_DIRECT_SYSTEM, MODIFY_DIRECT_USER
from .simulation_analysis import (
    SIMULATION_ANALYSIS_SYSTEM,
    get_simulation_analysis_prompt,
    get_simulation_analysis_system,
    get_compact_analysis_prompt,
)

__all__ = [
    'GENERATE_SYSTEM', 'GENERATE_USER',
    'GENERATE_DIRECT_SYSTEM', 'GENERATE_DIRECT_USER',
    'MODIFY_SYSTEM', 'MODIFY_USER',
    'MODIFY_DIRECT_SYSTEM', 'MODIFY_DIRECT_USER',
    'SIMULATION_ANALYSIS_SYSTEM',
    'get_simulation_analysis_prompt',
    'get_simulation_analysis_system',
    'get_compact_analysis_prompt',
]
