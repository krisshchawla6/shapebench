RESPONSE_FORMAT = """
You MUST respond with ONLY a valid JSON object (no markdown, no schema definitions).
Do NOT include $schema, title, description, type, properties, or required fields.
ALL floats MUST have exactly 8 decimal places (e.g., 0.98521047 not 0.9852).

Example response format:
{example_json}
"""

def format_response_instructions(example_json: str) -> str:
    return RESPONSE_FORMAT.format(example_json=example_json)
