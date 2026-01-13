"""
Milestone 3: End-to-end integration test
Tests complete main.py workflow (except final geometry generation)
"""
import os
import sys
import subprocess
from pathlib import Path

def test_full_integration():
    print("=== MILESTONE 3: Full Integration Test ===\n")
    
    # Cleanup previous test runs
    projects_dir = Path("projects/designs")  # Relative to test dir where main.py runs
    if projects_dir.exists():
        import shutil
        shutil.rmtree(projects_dir)
        print("[OK] Cleaned previous test data")
    
    # Run main.py
    cmd = [
        sys.executable,
        "../design_optimization/main.py",
        "../testing/plane_model.ntop",
        "Optimize for high-altitude flight with improved lift-to-drag ratio",
        "--variations", "3"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"\n[FAIL] Exit code {result.returncode}")
        return False
    
    # Validate outputs
    template = projects_dir / "input_template.json"
    assert template.exists(), "Template not created"
    print(f"[OK] Template exists: {template}")
    
    for i in range(1, 4):
        design_dir = projects_dir / f"design_{i}"
        input_json = design_dir / "input.json"
        assert design_dir.exists(), f"Design {i} folder not created"
        assert input_json.exists(), f"Design {i} input.json not created"
        print(f"[OK] Design {i} created: {input_json}")
    
    print("\n[OK] SUCCESS: Full integration working!")
    return True

if __name__ == "__main__":
    success = test_full_integration()
    sys.exit(0 if success else 1)
