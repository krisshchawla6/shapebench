"""
Milestone 2: Test design_agent variation generation
Tests that Gemini can generate intelligent variations from a template
"""
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "design_optimization"))
from design_agent import generate_variations

def test_variation_generation():
    print("=== MILESTONE 2: AI Variation Generation ===\n")
    
    # Use example template
    template_path = Path("../example_parametrization/input_template.json")
    assert template_path.exists(), "Example template not found"
    
    prompt = "Make the aircraft more aerodynamic and lightweight for long-range flight"
    num_variations = 3
    
    print(f"Template: {template_path}")
    print(f"Prompt: {prompt}")
    print(f"Variations: {num_variations}\n")
    
    try:
        variations = generate_variations(
            template_path=template_path,
            prompt=prompt,
            images_folder=None,
            num_variations=num_variations
        )
        
        if not variations:
            print("✗ FAILED: No variations generated")
            return False
        
        if len(variations) != num_variations:
            print(f"⚠ WARNING: Expected {num_variations}, got {len(variations)}")
        
        # Validate structure
        for i, var in enumerate(variations, 1):
            assert "inputs" in var, f"Variation {i} missing 'inputs'"
            assert "title" in var, f"Variation {i} missing 'title'"
            print(f"✓ Variation {i} structure valid")
        
        print(f"\n✓ SUCCESS: Generated {len(variations)} valid variations")
        
        # Save for inspection
        output_file = Path("test_variations.json")
        with open(output_file, 'w') as f:
            json.dump(variations, f, indent=2)
        print(f"✓ Saved to {output_file} for inspection")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_variation_generation()
    sys.exit(0 if success else 1)
