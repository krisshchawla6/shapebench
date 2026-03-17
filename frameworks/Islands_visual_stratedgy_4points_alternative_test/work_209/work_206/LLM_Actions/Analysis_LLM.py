"""
Simulation Analysis LLM Agent

Post-processes CFD simulation results using Gemini to generate critical,
concise analysis based on flow field images and quantitative metrics.

Usage:
    from prompts.Analysis_LLM import run_simulation_analysis
    
    # Prepare inputs
    images = ['path/to/1_p.png', 'path/to/1_u.png', 'path/to/1_v.png']
    metrics = {
        'drag': 0.0234,
        'lift': 0.156,
        'reward': 2.34
    }
    
    # Run analysis
    analysis_text = run_simulation_analysis(images, metrics)
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Import prompts - handle both module and direct script execution
try:
    from prompts.simulation_analysis import (
        get_simulation_analysis_prompt,
        get_simulation_analysis_system,
        get_compact_analysis_prompt
    )
except ImportError:
    # Fallback for different execution contexts
    try:
        from .prompts.simulation_analysis import (
            get_simulation_analysis_prompt,
            get_simulation_analysis_system,
            get_compact_analysis_prompt
        )
    except ImportError:
         from modified_env.LLM_Actions.prompts.simulation_analysis import (
            get_simulation_analysis_prompt,
            get_simulation_analysis_system,
            get_compact_analysis_prompt
        )

# Load environment variables
_dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(_dotenv_path)

_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if _api_key:
    genai.configure(api_key=_api_key)
else:
    print("Warning: Google API Key not found for Analysis_LLM.")


def _load_images(image_paths: List[str]) -> List[Any]:
    """Load images from paths, returning PIL Image objects.
    
    Args:
        image_paths: List of paths to PNG files
        
    Returns:
        List of loaded PIL Image objects
    """
    images = []
    for path in image_paths:
        if os.path.exists(path):
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {path}: {e}")
        else:
            print(f"Warning: Image not found: {path}")
    return images


def _extract_analysis(text: str) -> Optional[str]:
    """Extract the analysis section from LLM response.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Extracted analysis text, or None if not found
    """
    # Try to extract structured <ANALYSIS> block
    analysis_match = re.search(r'<ANALYSIS>(.*?)</ANALYSIS>', text, re.DOTALL)
    if analysis_match:
        return analysis_match.group(1).strip()
    
    # Fallback: try to find paragraphs with headers
    # Look for the typical structure with **PARAGRAPH N:**
    paragraph_pattern = r'\*\*(?:PARAGRAPH\s*\d+|Performance|Critical|Recommendation)[:\s].*?(?=\*\*(?:PARAGRAPH|Performance|Critical|Recommendation)|$)'
    paragraphs = re.findall(paragraph_pattern, text, re.DOTALL | re.IGNORECASE)
    if paragraphs:
        return '\n\n'.join(p.strip() for p in paragraphs)
    
    # Last resort: return the full text if it's reasonably short
    if len(text) < 2000:
        return text.strip()
    
    return None


def _extract_reasoning(text: str) -> Optional[str]:
    """Extract the reasoning/chain-of-thought section from LLM response.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Extracted reasoning text, or None if not found
    """
    reasoning_match = re.search(r'<REASONING>(.*?)</REASONING>', text, re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    return None


def run_simulation_analysis(
    image_paths: List[str],
    metrics: Dict[str, Any],
    temperature: float = 0.7,
    compact: bool = False,
    return_reasoning: bool = False,
    debug: bool = False
) -> str | Tuple[str, Optional[str]]:
    """Run Gemini analysis on simulation results.
    
    Args:
        image_paths: List of paths to flow field images (p, u, v PNG files)
        metrics: Dictionary of quantitative values with keys like:
            - 'drag': Drag coefficient (float)
            - 'lift': Lift coefficient (float)
            - 'reward': Objective function value (float)
            - Any additional metrics to include
        temperature: LLM sampling temperature (lower = more focused)
        compact: If True, use compact prompt variant
        return_reasoning: If True, return (analysis, reasoning) tuple
        debug: If True, print debug information
        
    Returns:
        Analysis text string, or (analysis, reasoning) tuple if return_reasoning=True
    """
    # Validate inputs
    if not image_paths:
        raise ValueError("At least one image path must be provided")
    if not metrics:
        raise ValueError("Metrics dictionary cannot be empty")
    
    # Load images
    images = _load_images(image_paths)
    if not images:
        raise ValueError("No valid images could be loaded")
    
    # Build prompts
    system_prompt = get_simulation_analysis_system()
    if compact:
        user_prompt = get_compact_analysis_prompt(metrics)
    else:
        user_prompt = get_simulation_analysis_prompt(metrics)
    
    if debug:
        print("=" * 60)
        print("SYSTEM PROMPT:")
        print(system_prompt)
        print("=" * 60)
        print("USER PROMPT:")
        print(user_prompt)
        print("=" * 60)
        print(f"Loaded {len(images)} images")
    
    try:
        # Initialize model
        model = genai.GenerativeModel(
            'gemini-3-pro-preview',
            system_instruction=system_prompt
        )
        
        # Build content with images + prompt
        content = [user_prompt] + images
        
        # Generate response
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content(content, generation_config=generation_config)
        raw_text = response.text
        
        if debug:
            print("RAW RESPONSE:")
            print(raw_text)
            print("=" * 60)
        
        # Extract analysis
        analysis = _extract_analysis(raw_text)
        
        if analysis is None:
            print("Warning: Could not extract structured analysis, returning raw response")
            analysis = raw_text
        
        if return_reasoning:
            reasoning = _extract_reasoning(raw_text)
            return analysis, reasoning
        
        return analysis
        
    except Exception as e:
        error_msg = f"Analysis LLM Error: {e}"
        print(error_msg)
        if return_reasoning:
            return error_msg, None
        return error_msg


def run_batch_analysis(
    results: List[Dict],
    output_dir: Optional[str] = None,
    temperature: float = 0.7
) -> List[Dict]:
    """Run analysis on multiple simulation results.
    
    Args:
        results: List of result dictionaries, each containing:
            - 'images': List of image paths
            - 'metrics': Dictionary of quantitative values
            - Optionally 'name': Identifier for this result
        output_dir: If provided, save analyses to this directory
        temperature: LLM sampling temperature
        
    Returns:
        List of dictionaries with 'name', 'analysis', and 'reasoning' keys
    """
    analyses = []
    
    for i, result in enumerate(results):
        name = result.get('name', f'result_{i}')
        images = result.get('images', [])
        metrics = result.get('metrics', {})
        
        print(f"Analyzing {name}...")
        
        try:
            analysis, reasoning = run_simulation_analysis(
                images, metrics,
                temperature=temperature,
                return_reasoning=True
            )
            
            entry = {
                'name': name,
                'analysis': analysis,
                'reasoning': reasoning,
                'metrics': metrics
            }
            analyses.append(entry)
            
            # Save to file if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                analysis_path = os.path.join(output_dir, f'{name}_analysis.txt')
                with open(analysis_path, 'w') as f:
                    f.write(f"# Analysis for {name}\n\n")
                    f.write(f"## Metrics\n")
                    for k, v in metrics.items():
                        f.write(f"- {k}: {v}\n")
                    f.write(f"\n## Analysis\n{analysis}\n")
                    if reasoning:
                        f.write(f"\n## Reasoning\n{reasoning}\n")
                        
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            analyses.append({
                'name': name,
                'analysis': f"Error: {e}",
                'reasoning': None,
                'metrics': metrics
            })
    
    return analyses


# =============================================================================
# CONVENIENCE FUNCTION FOR INTEGRATION
# =============================================================================

def analyze_simulation_result(
    sol_dir: str,
    drag: float,
    lift: float,
    reward: float,
    **extra_metrics
) -> str:
    """Convenience function to analyze a simulation from standard output directory.
    
    Args:
        sol_dir: Path to solution directory containing 1_p.png, 1_u.png, 1_v.png
        drag: Drag coefficient
        lift: Lift coefficient  
        reward: Objective function value
        **extra_metrics: Any additional metrics to include
        
    Returns:
        Analysis text string
    """
    # Build image paths
    image_paths = [
        os.path.join(sol_dir, '1_p.png'),
        os.path.join(sol_dir, '1_u.png'),
        os.path.join(sol_dir, '1_v.png')
    ]
    
    # Build metrics
    metrics = {
        'drag': drag,
        'lift': lift,
        'reward': reward,
        **extra_metrics
    }
    
    return run_simulation_analysis(image_paths, metrics)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simulation analysis with Gemini')
    parser.add_argument('--sol-dir', type=str, required=True,
                        help='Path to solution directory with flow field images')
    parser.add_argument('--drag', type=float, required=True, help='Drag coefficient')
    parser.add_argument('--lift', type=float, required=True, help='Lift coefficient')
    parser.add_argument('--reward', type=float, required=True, help='Reward value')
    parser.add_argument('--output', type=str, help='Output file path for analysis')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--compact', action='store_true', help='Use compact prompt')
    
    args = parser.parse_args()
    
    # Build image paths
    image_paths = [
        os.path.join(args.sol_dir, '1_p.png'),
        os.path.join(args.sol_dir, '1_u.png'),
        os.path.join(args.sol_dir, '1_v.png')
    ]
    
    metrics = {
        'drag': args.drag,
        'lift': args.lift,
        'reward': args.reward
    }
    
    print(f"Analyzing results from {args.sol_dir}...")
    print(f"Metrics: drag={args.drag}, lift={args.lift}, reward={args.reward}")
    
    analysis, reasoning = run_simulation_analysis(
        image_paths, metrics,
        compact=args.compact,
        return_reasoning=True,
        debug=args.debug
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print(analysis)
    
    if reasoning:
        print("\n" + "=" * 60)
        print("REASONING:")
        print("=" * 60)
        print(reasoning)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(analysis)
        print(f"\nAnalysis saved to {args.output}")
