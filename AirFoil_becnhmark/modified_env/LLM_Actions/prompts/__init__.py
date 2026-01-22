# Airfoil Design LLM Prompts
from .base import CONTEXT_FORMAT, format_context
from .generate import GENERATE_SYSTEM, GENERATE_USER
from .generate_direct import GENERATE_DIRECT_SYSTEM, GENERATE_DIRECT_USER
from .modify import MODIFY_SYSTEM, MODIFY_USER
from .modify_direct import MODIFY_DIRECT_SYSTEM, MODIFY_DIRECT_USER

__all__ = [
    'CONTEXT_FORMAT', 'format_context',
    'GENERATE_SYSTEM', 'GENERATE_USER',
    'GENERATE_DIRECT_SYSTEM', 'GENERATE_DIRECT_USER', 
    'MODIFY_SYSTEM', 'MODIFY_USER',
    'MODIFY_DIRECT_SYSTEM', 'MODIFY_DIRECT_USER',
]
