import os
import json
import base64
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv


def generate_variations(template_path, prompt, images_folder=None, num_variations=5):
    """
    Uses Gemini to generate intelligent design variations based on the template and user prompt.
    """
    load_dotenv()
    api_key = os.getenv("gemini_key")
    if not api_key:
        raise ValueError("Error: Set gemini_key in .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    # Load base template
    with open(template_path, 'r') as f:
        base_template = json.load(f)

    # Prepare image context if provided
    image_parts = []
    if images_folder:
        img_dir = Path(images_folder)
        if img_dir.exists():
            for img_file in img_dir.glob("*.png"):
                image_parts.append(genai.upload_file(str(img_file)))
                print(f"[OK] Loaded image: {img_file.name}")

    # Build intelligent prompt for Gemini
    system_prompt = f"""You are an expert engineering design optimization AI specializing in parametric CAD and design space exploration.

**Base Design Template:**
```json
{json.dumps(base_template, indent=2)}
```

**User's Design Goal:**
{prompt}

**Task:**
Generate {num_variations} distinct, engineering-valid design variations that intelligently modify the parameters above to explore the design space effectively.

**Requirements:**
1. Each variation must maintain structural/engineering validity
2. Parameter changes should be meaningful and purposeful (not random)
3. Explore diverse regions of the design space
4. DO NOT modify parameters of type "file_path" or "text" (like NACA codes) unless specifically requested
5. Focus on "real" type parameters (dimensions, sizes, etc.)
6. Provide brief reasoning for each design variation

**Output Format:**
Return a JSON array with {num_variations} objects. Each object must have:
- "reasoning": Brief explanation of parameter choices
- "schema": Complete modified JSON matching the input template structure

Example:
```json
[
  {{
    "reasoning": "Increased wing span for better lift-to-drag ratio while reducing fuselage width for weight savings",
    "schema": {{ ... complete modified template ... }}
  }},
  ...
]
```

Generate the variations now:"""

    # Call Gemini with or without images
    try:
        if image_parts:
            response = model.generate_content([system_prompt] + image_parts)
        else:
            response = model.generate_content(system_prompt)

        # Parse response
        response_text = response.text.strip()
        
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        variations_data = json.loads(response_text)

        # Extract and validate schemas
        variations = []
        for i, var in enumerate(variations_data, 1):
            schema = var.get("schema", {})
            reasoning = var.get("reasoning", "No reasoning provided")
            
            # Validate structure matches base template
            if "inputs" in schema and "title" in schema:
                print(f"\n[OK] Variation {i}: {reasoning}")
                variations.append(schema)
            else:
                print(f"[FAIL] Variation {i} invalid structure, skipping")

        return variations

    except Exception as e:
        print(f"Error generating variations: {e}")
        return []
