# Simulation Analysis Prompts
# Post-processing and critical analysis of CFD simulation results

from typing import Dict, List, Optional

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SIMULATION_ANALYSIS_SYSTEM = """You are an expert aerodynamicist. Analyze CFD results and provide insights for shape optimization."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

SIMULATION_ANALYSIS_USER = """## Metrics
{metrics_section}

## Images
Pressure field, u-velocity, v-velocity visualizations provided.

## Task
Goal: MAXIMIZE reward. Analyze the flow field images and explain what you observe.

<THINKING>
What do you see in the flow fields? How does the geometry cause these flow features?
</THINKING>

<ANALYSIS>
Describe the flow behavior and how the shape affects it. What geometric changes would improve the flow and increase reward?
</ANALYSIS>"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_metrics_section(metrics: Dict) -> str:
    """Format the quantitative metrics dictionary into a readable section.
    
    Args:
        metrics: Dictionary containing quantitative values like:
            - drag: float
            - lift: float  
            - reward: float
            - ld_ratio: float (optional, will compute if missing)
            - Any other custom metrics
            
    Returns:
        Formatted metrics string
    """
    lines = []
    
    # Extract standard metrics
    # drag = metrics.get('drag', 0.0)
    # lift = metrics.get('lift', 0.0)
    reward = metrics.get('reward', 0.0)
    
    # Format core metrics - minimal, let LLM interpret
    # lines.append(f"- Drag: {drag:.4f}")
    # lines.append(f"- Lift: {lift:.4f}")
    # lines.append(f"- L/D: {ld_ratio:.4f}" if ld_ratio != float('inf') else "- L/D: inf")
    lines.append(f"- **Reward: {reward:.4f}** (MAXIMIZE THIS)")
    
    # Add any additional metrics (excluding drag/lift)
    standard_keys = {'drag', 'lift', 'reward', 'ld_ratio'}
    for key, value in metrics.items():
        if key not in standard_keys:
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
    
    return "\n".join(lines)


def get_simulation_analysis_prompt(metrics: Dict) -> str:
    """Build the complete simulation analysis prompt.
    
    Args:
        metrics: Dictionary of quantitative values
        
    Returns:
        Complete formatted user prompt
    """
    metrics_section = format_metrics_section(metrics)
    return SIMULATION_ANALYSIS_USER.format(metrics_section=metrics_section)


def get_simulation_analysis_system() -> str:
    """Get the system prompt for simulation analysis.
    
    Returns:
        System prompt string
    """
    return SIMULATION_ANALYSIS_SYSTEM


# =============================================================================
# COMPACT VARIANT (for token-limited contexts)
# =============================================================================

SIMULATION_ANALYSIS_COMPACT_USER = """## Metrics
{metrics_section}

## Images
Pressure, u-velocity, v-velocity fields provided.

Analyze the flow and geometry. What do you observe? What should change?"""


def get_compact_analysis_prompt(metrics: Dict) -> str:
    """Build a more compact analysis prompt for token-limited scenarios.
    
    Args:
        metrics: Dictionary of quantitative values
        
    Returns:
        Compact formatted user prompt
    """
    metrics_section = format_metrics_section(metrics)
    return SIMULATION_ANALYSIS_COMPACT_USER.format(metrics_section=metrics_section)
