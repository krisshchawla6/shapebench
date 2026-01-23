# Base prompts and context formatting for airfoil design
# Restructured following Shinka's best practices

from typing import List, Dict, Optional

# =============================================================================
# CONTEXT TEMPLATES
# =============================================================================

CONTEXT_FORMAT = """## Parameter Space

N control points, each with 3 params in [-1.0, 1.0]:
- radius_param: radial distance from center
- angle_param: angular offset
- edgy_param: corner sharpness (0=smooth, 1=sharp)

Action: [r0, a0, e0, r1, a1, e1, ..., rN, aN, eN]

## Objective: MAXIMIZE reward. Higher = better. No upper limit.

{parent_section}

{inspirations_section}
"""

CONTEXT_FORMAT_MINIMAL = """## Params: N control points × 3 values (radius, angle, edgy) in [-1,1]

## Objective: MAXIMIZE reward

{parent_section}

{inspirations_section}
"""

# Parent design section - for all actions
PARENT_DESIGN_SECTION = """## Reference Design

- **Parameters**: {action_vector}
- **Reward**: {reward}
- **Rank**: #{rank}
{feedback_section}

Note: This shows one design that was tested. Explore different parameter combinations to maximize reward."""

# Inspiration designs section 
INSPIRATION_DESIGNS_SECTION = """## Additional Reference Designs

{inspiration_entries}"""

INSPIRATION_ENTRY_DETAILED = """### Design {idx} (Rank #{rank})
- **Parameters**: {action_vector}
- **Reward**: {reward}
{feedback_section}"""

# Legacy templates (kept for backwards compatibility)
BEST_DESIGN_SECTION = """## Best Known Design

The best design so far achieved a reward of **{best_reward:.4f}**:
- Action: {best_action}
- Drag: {best_drag:.4f}
- Lift: {best_lift:.4f}
- L/D Ratio: {best_ld_ratio}
{best_feedback}"""

DESIGN_ENTRY_DETAILED = """### Design {idx} {rank_badge}
- **Action**: {action_vector}
- **Reward**: {reward:.4f}
- **Drag**: {drag:.4f}
- **Lift**: {lift:.4f}
- **L/D Ratio**: {ld_ratio}
- **Rank**: #{rank}
{feedback_section}"""

DESIGN_ENTRY_COMPACT = """Design {idx}: Reward={reward:.4f}, Drag={drag:.4f}, Lift={lift:.4f}, Rank=#{rank}
  Action: {action_vector}{feedback_section}"""

RESPONSE_FORMAT = """
You MUST respond with ONLY a valid JSON object (no markdown, no schema definitions).
Do NOT include $schema, title, description, type, properties, or required fields.

Example response format:
{example_json}
"""


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_reward(reward: float) -> str:
    """Format reward with context about goal (MAXIMIZE)."""
    if reward > 1.0:
        return f"{reward:.4f} (decent - push higher)"
    elif reward > 0.1:
        return f"{reward:.4f} (low - much room to improve)"
    elif reward > 0:
        return f"{reward:.4f} (very low - needs major improvement)"
    elif reward > -1:
        return f"{reward:.4f} (negative - underperforming)"
    elif reward > -5:
        return f"{reward:.4f} (poor - significant issues)"
    else:
        return f"{reward:.4f} (very poor - fundamental problems)"


# TODO: Replace with separate qualitative agent
# def analyze_design_characteristics(action_vector: List[float]) -> str:
#     """Analyze and describe key characteristics of a design."""
#     if not action_vector or len(action_vector) < 3:
#         return "Unknown"
#     
#     n_params = len(action_vector)
#     n_control_points = n_params // 3
#     
#     # Extract parameter groups
#     radii = [action_vector[i*3] for i in range(n_control_points)]
#     angles = [action_vector[i*3 + 1] for i in range(n_control_points)]
#     edginess = [action_vector[i*3 + 2] for i in range(n_control_points)]
#     
#     characteristics = []
#     
#     # Analyze radii
#     avg_radius = sum(radii) / len(radii)
#     if avg_radius > 0.5:
#         characteristics.append("large overall size")
#     elif avg_radius < -0.5:
#         characteristics.append("compact shape")
#     
#     # Analyze angles
#     angle_variance = sum((a - sum(angles)/len(angles))**2 for a in angles) / len(angles)
#     if angle_variance > 0.3:
#         characteristics.append("asymmetric shape")
#     else:
#         characteristics.append("relatively symmetric")
#     
#     # Analyze edginess
#     avg_edgy = sum(edginess) / len(edginess)
#     if avg_edgy > 0.3:
#         characteristics.append("sharp edges")
#     elif avg_edgy < -0.3:
#         characteristics.append("smooth curves")
#     
#     return ", ".join(characteristics) if characteristics else "balanced parameters"

def analyze_design_characteristics(action_vector: List[float]) -> str:
    """Placeholder - to be replaced with separate qualitative agent."""
    return ""  # Will be filled by qualitative agent


def format_feedback_section(feedback: Optional[str]) -> str:
    """Format text feedback for inclusion in design entry."""
    if not feedback or not feedback.strip():
        return ""
    return f"\n- **Feedback**: {feedback.strip()}"


def format_context(
    context: List[Dict],
    detailed: bool = True,
    include_feedback: bool = True,
    max_designs: int = 10,
    highlight_best: bool = True
) -> str:
    """Format design history for prompts.
    
    Context structure:
        - context[0] = Parent design (the design being modified/built upon)
        - context[1:] = Inspiration designs (additional reference designs)
    
    Args:
        context: List of design dictionaries with 'vector', 'reward', 'ranking', 'feedback' keys
                 First item is the PARENT, rest are INSPIRATIONS
        detailed: If True, use detailed format; otherwise compact
        include_feedback: Whether to include text feedback if available
        max_designs: Maximum number of designs to include
        highlight_best: Ignored (kept for backwards compatibility)
        
    Returns:
        Formatted context string
    """
    if not context:
        return CONTEXT_FORMAT_MINIMAL.format(
            parent_section="No parent design available. This is the first design!",
            inspirations_section=""
        )
    
    # =================================================================
    # Format PARENT design (first item in context)
    # =================================================================
    parent = context[0]
    parent_reward = parent.get('reward', 0.0)
    parent_rank = parent.get('ranking', 0)
    parent_action = parent.get('vector', [])
    parent_feedback = parent.get('feedback', '') if include_feedback else ''
    parent_drag = parent.get('drag', 0.0)
    parent_lift = parent.get('lift', 0.0)
    parent_ld_ratio = f"{parent_lift / abs(parent_drag):.4f}" if parent_drag != 0 else "∞"
    
    feedback_section = format_feedback_section(parent_feedback)
    
    parent_section = PARENT_DESIGN_SECTION.format(
        action_vector=parent_action,
        reward=format_reward(parent_reward),
        rank=parent_rank,
        feedback_section=feedback_section
    )
    
    # =================================================================
    # Format INSPIRATION designs (remaining items in context)
    # =================================================================
    inspirations_section = ""
    if len(context) > 1:
        inspiration_lines = []
        
        # Track seen designs to avoid duplicates
        seen = {tuple(parent_action) if isinstance(parent_action, list) else parent_action}
        
        for i, item in enumerate(context[1:max_designs], start=1):
            reward = item.get('reward', 0.0)
            rank = item.get('ranking', i)
            action_vector = item.get('vector', [])
            drag = item.get('drag', 0.0)
            lift = item.get('lift', 0.0)
            ld_ratio = f"{lift / abs(drag):.4f}" if drag != 0 else "∞"
            feedback = item.get('feedback', '') if include_feedback else ''
            
            # Skip duplicates
            action_key = tuple(action_vector) if isinstance(action_vector, list) else action_vector
            if action_key in seen:
                continue
            seen.add(action_key)
            
            feedback_section = format_feedback_section(feedback)
            
            inspiration_lines.append(INSPIRATION_ENTRY_DETAILED.format(
                idx=i,
                action_vector=action_vector,
                reward=format_reward(reward),
                rank=rank,
                feedback_section=feedback_section
            ))
        
        if inspiration_lines:
            inspirations_section = INSPIRATION_DESIGNS_SECTION.format(
                inspiration_entries='\n'.join(inspiration_lines)
            )
    
    if detailed:
        return CONTEXT_FORMAT.format(
            parent_section=parent_section,
            inspirations_section=inspirations_section
        )
    else:
        return CONTEXT_FORMAT_MINIMAL.format(
            parent_section=parent_section,
            inspirations_section=inspirations_section
        )


def format_response_instructions(example_json: str) -> str:
    """Format response instructions with example."""
    return RESPONSE_FORMAT.format(example_json=example_json)


# =============================================================================
# TEXT FEEDBACK HELPERS (following Shinka pattern)
# =============================================================================

def format_text_feedback_section(text_feedback: Optional[str]) -> str:
    """Format text feedback for inclusion in prompts (Shinka-style)."""
    if not text_feedback or not text_feedback.strip():
        return ""
    
    feedback_text = text_feedback
    if isinstance(feedback_text, list):
        feedback_text = "\n".join(feedback_text)
    
    return f"""
## Additional Feedback

{feedback_text.strip()}
"""


# TODO: Replace with separate qualitative agent
# def construct_design_analysis_msg(
#     designs: List[Dict],
#     include_feedback: bool = True
# ) -> str:
#     """Construct a detailed analysis message for designs (Shinka-style).
#     
#     Args:
#         designs: List of design dictionaries
#         include_feedback: Whether to include text feedback
#         
#     Returns:
#         Formatted analysis string
#     """
#     if not designs:
#         return "No designs to analyze."
#     
#     analysis_str = "## Performance Analysis of Previous Designs\n\n"
#     
#     # Sort by performance
#     sorted_designs = sorted(designs, key=lambda x: x.get('reward', float('-inf')), reverse=True)
#     
#     for i, design in enumerate(sorted_designs):
#         reward = design.get('reward', 0.0)
#         action = design.get('vector', [])
#         
#         analysis_str += f"### Rank #{i + 1}: Reward = {reward:.4f}\n"
#         analysis_str += f"- Parameters: {action}\n"
#         analysis_str += f"- Characteristics: {analyze_design_characteristics(action)}\n"
#         
#         if include_feedback and design.get('feedback'):
#             analysis_str += f"- Feedback: {design['feedback']}\n"
#         
#         analysis_str += "\n"
#     
#     return analysis_str

def construct_design_analysis_msg(
    designs: List[Dict],
    include_feedback: bool = True
) -> str:
    """Placeholder - to be replaced with separate qualitative agent."""
    return ""  # Will be filled by qualitative agent


# =============================================================================
# LEGACY SUPPORT
# =============================================================================

# Keep old templates for backwards compatibility
DESIGN_ENTRY = """Design {idx}:
  - Action: {action_vector}
  - Reward: {reward:.4f}
  - Rank: {rank}
"""
