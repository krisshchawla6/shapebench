import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
from design_agent import generate_variations


def main():
    parser = argparse.ArgumentParser(description="AI-powered design variation generator for nTop")
    parser.add_argument("ntop_file", help="Path to .ntop file")
    parser.add_argument("prompt", help="Design optimization prompt")
    parser.add_argument("--images", help="Optional folder containing design images", default=None)
    parser.add_argument("--variations", type=int, default=5, help="Number of design variations to generate")
    args = parser.parse_args()

    load_dotenv()
    ntopcl_path = os.getenv("ntopcl_path")
    username = os.getenv("ntop_username")
    password = os.getenv("ntop_password")

    if not ntopcl_path:
        print("Error: Set ntopcl_path in .env")
        sys.exit(1)

    ntop_file = Path(args.ntop_file)
    if not ntop_file.exists():
        print(f"Error: .ntop file not found: {ntop_file}")
        sys.exit(1)

    # Phase 1: Setup and Template Generation
    print(f"\n=== Phase 1: Generating Template from {ntop_file.name} ===")
    projects_dir = Path(r"C:\Users\kriss\Desktop\NTop\projects\designs")
    projects_dir.mkdir(parents=True, exist_ok=True)

    # Generate template using ntopcl -t
    cmd = [ntopcl_path]
    if username and password:
        cmd.extend(["-u", username, "-w", password])
    cmd.extend(["-t", str(ntop_file)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating template: {result.stderr}")
        sys.exit(1)

    # Move template to projects/designs/
    template_src = Path("input_template.json")
    template_dst = projects_dir / "input_template.json"
    
    if not template_src.exists():
        print("Error: input_template.json not generated")
        sys.exit(1)
    
    if template_dst.exists():
        template_dst.unlink()
    template_src.rename(template_dst)
    print(f"[OK] Template saved to {template_dst}")

    # Phase 2: Generate Design Variations
    print(f"\n=== Phase 2: Generating {args.variations} Design Variations ===")
    variations = generate_variations(
        template_path=template_dst,
        prompt=args.prompt,
        images_folder=args.images,
        num_variations=args.variations
    )

    if not variations:
        print("Error: No variations generated")
        sys.exit(1)

    # Save each variation with correct file paths
    for i, variation in enumerate(variations, 1):
        design_dir = projects_dir / f"design_{i}"
        design_dir.mkdir(exist_ok=True)

        # Update file_path parameter to point to design-specific folder
        for param in variation.get("inputs", []):
            if param.get("type") == "file_path":
                param["value"] = f"C:/Users/kriss/Desktop/NTop/projects/designs/design_{i}/mesh_output.stl"

        variation_path = design_dir / "input.json"
        with open(variation_path, 'w') as f:
            json.dump(variation, f, indent=4)
        
        print(f"[OK] Design {i} saved to {variation_path}")

    # Phase 3: Execute Design Generation
    print(f"\n=== Phase 3: Generating Geometries ===")
    for i in range(1, args.variations + 1):
        design_dir = projects_dir / f"design_{i}"
        input_json = design_dir / "input.json"
        output_json = design_dir / "output.json"

        cmd = [ntopcl_path]
        if username and password:
            cmd.extend(["-u", username, "-w", password])
        cmd.extend(["-v", "2", "-j", str(input_json), "-o", str(output_json), str(ntop_file)])

        print(f"\n[Design {i}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[FAIL] Design {i} failed: {result.stderr}")
            continue

        print(f"[OK] Design {i} generated successfully")
        if result.stdout:
            print(result.stdout.strip())

    print("\n=== All Done! ===")
    print(f"Generated {args.variations} design variations in {projects_dir}/")


if __name__ == "__main__":
    main()
