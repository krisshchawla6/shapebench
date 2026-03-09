# Thin compatibility shim -- the strategy prompt files import format_response_instructions
# from here. The actual env-specific context formatting comes from the environment's
# prompt_blocks at runtime, but this format is universal.

RESPONSE_FORMAT = """
You MUST respond with ONLY a valid JSON object (no markdown, no schema definitions).
Do NOT include $schema, title, description, type, properties, or required fields.

Example response format:
{example_json}
"""


def format_response_instructions(example_json: str) -> str:
    """Format response instructions with example."""
    return RESPONSE_FORMAT.format(example_json=example_json)
