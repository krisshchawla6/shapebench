"""
Milestone 1: Test template generation phase
Tests that ntopcl -t runs successfully and generates input_template.json
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def test_template_generation():
    print("=== MILESTONE 1: Template Generation ===\n")
    
    # Setup
    ntopcl_path = os.getenv("ntopcl_path")
    username = os.getenv("ntop_username")
    password = os.getenv("ntop_password")
    ntop_file = Path("../testing/plane_model.ntop")
    
    assert ntopcl_path, "ntopcl_path not in .env"
    assert ntop_file.exists(), f".ntop file not found: {ntop_file}"
    
    print(f"✓ ntopcl_path: {ntopcl_path}")
    print(f"✓ ntop_file: {ntop_file}")
    
    # Run template generation
    cmd = [ntopcl_path]
    if username and password:
        cmd.extend(["-u", username, "-w", password])
    cmd.extend(["-t", str(ntop_file)])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ FAILED: {result.stderr}")
        return False
    
    # Check template was created
    template_file = Path("input_template.json")
    if not template_file.exists():
        print("✗ FAILED: input_template.json not created")
        return False
    
    print(f"✓ SUCCESS: Template generated")
    print(f"✓ File size: {template_file.stat().st_size} bytes")
    
    # Cleanup
    template_file.unlink()
    if Path("output_template.json").exists():
        Path("output_template.json").unlink()
    
    return True

if __name__ == "__main__":
    success = test_template_generation()
    sys.exit(0 if success else 1)
