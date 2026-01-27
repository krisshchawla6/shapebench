"""
LLM PSO prompt - Minimalist evolutionary optimizer format
"""

from typing import List, Dict


LLM_PSO_SYSTEM = """Role: You are an evolutionary optimizer.

Task: You will MAXIMIZE a function F[r, angle, e] with 3 variables r, angle, e.

Range: r, angle, e range between [-1, 1]."""


def format_pso_records(context: List[Dict]) -> str:
    """Format context into PSO records format.
    
    Args:
        context: List of design dicts with 'vector', 'reward', 'ranking'
        
    Returns:
        Formatted records string
    """
    records = []
    
    for item in context:
        reward = item.get('reward', 0.0)
        vector = item.get('vector', [])
        rank = item.get('ranking', 0)
        
        # Format vector as [r, angle, e, r, angle, e, ...]
        # Show only first 3 values per control point for brevity
        formatted_values = []
        for i in range(0, len(vector), 3):
            if i+2 < len(vector):
                formatted_values.append(f"[{vector[i]:.3f}, {vector[i+1]:.3f}, {vector[i+2]:.3f}]")
        
        # Format: reward: [values]; (rank #N)
        record = f"{reward:.6f}: {', '.join(formatted_values)} (rank #{rank})"
        records.append(record)
    
    return '\n'.join(records)


LLM_PSO_USER_TEMPLATE = """Records: Below are the records of the top performing generations of [r, angle, e] values. Each line shows the reward and corresponding parameter values. Different generations are sorted by their objective function values F[r, angle, e].

{records}

Determine a new [r, angle, e] value for the next generation to achieve higher F[r, angle, e] values.

Format your output as [r, angle, e]. No explanation needed."""


def get_pso_prompt(context: List[Dict]) -> str:
    """Build the LLM PSO prompt.
    
    Args:
        context: List of design dictionaries
        
    Returns:
        Formatted user prompt
    """
    records = format_pso_records(context)
    return LLM_PSO_USER_TEMPLATE.format(records=records)


def get_pso_system() -> str:
    """Get system prompt for LLM PSO."""
    return LLM_PSO_SYSTEM
