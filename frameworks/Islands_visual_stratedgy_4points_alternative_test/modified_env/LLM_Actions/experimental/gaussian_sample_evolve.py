import os, sys, json, numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict

# Setup
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PARENT_DIR, '.env'))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_KEY"))

def sample_offspring(inspirations: np.ndarray, m: int, n_cp: int) -> List[Dict]:
    mean, std = inspirations.mean(axis=0), inspirations.std(axis=0) + 0.04
    offspring = []
    for i in range(m):
        vec = np.random.normal(mean, std)
        vec[2::4] = np.clip(vec[2::4], 0, 1) # radius
        vec[3::4] = np.clip(vec[3::4], 0, 1) # edgy
        offspring.append({'id': i, 'vector': vec.tolist()})
    return offspring

def run_evolution(inspirations: List[Dict], parent: List[float] = None, m: int = 5, n_cp: int = 4):
    insp_vecs = np.array([i['vector'] for i in inspirations])
    offspring = sample_offspring(insp_vecs, m, n_cp)
    
    # Format Records: Objective:[vec]; [vec]...[best_vec]
    records = ""
    for item in sorted(inspirations, key=lambda x: x['reward']):
        v_str = f"[{','.join([f'{v:.3f}' for v in item['vector']])}]"
        records += f"{item['reward']:.6f}:{v_str}; {v_str}\n"

    candidates = "\n".join([f"ID {o['id']}: [{','.join([f'{v:.3f}' for v in o['vector']])}]" for o in offspring])

    prompt = f"""Role: You are an evolutionary optimizer.
Task: You will MAXIMIZE a function F[params]
Range: Parameters range between [-1.5, 1.5].
Records: Below are the records of the top performing generations of parameter values. Each line represents one generation sorted in ascending order of best objective value F.
{records}
Candidates:
{candidates}

Determine a new [params] value for the next generation to achieve higher F values by choosing the best ID from the candidates.
Format your output in [params]. No explanation needed."""

    res = genai.GenerativeModel('gemini-3-pro-preview').generate_content(prompt).text
    try:
        # Extract the vector from [v1, v2, ...]
        vec_str = res[res.find('['):res.rfind(']')+1]
        chosen_vec = json.loads(vec_str)
        # Find closest offspring to the LLM output
        best_id = np.argmin([np.linalg.norm(np.array(o['vector']) - np.array(chosen_vec)) for o in offspring])
        return offspring[best_id]
    except:
        return offspring[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspirations", type=str, required=True) # JSON list of {'vector':[], 'reward':float}
    parser.add_argument("--parent", type=str, default="[]")
    parser.add_argument("-m", type=int, default=5)
    args = parser.parse_args()
    
    res = run_evolution(json.loads(args.inspirations), json.loads(args.parent), m=args.m)
    print(json.dumps(res))
