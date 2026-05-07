"""Reward: maximise –fuel_mass (minimise fuel mass in kg).

Constraint (Table 2, Saves et al. 2022):
    0.05 < static_margin < 0.10

Violations are penalised with a linear coefficient so the optimiser
receives gradient signal toward the feasible region.

The paper's stated 10^-3 constraint tolerance is preserved: violations
smaller than CONSTRAINT_TOLERANCE are not penalised, consistent with the
reported paper optimum (sm=0.0495) being accepted despite nominally violating
the lower bound by 0.0005.

Designs with unavailable or NaN static_margin are returned as FAIL_REWARD —
the paper discards any failed MDA evaluation without adding it to the sample set.

Note: the paper uses hard constraints via SEGO/UTB surrogate models. A scalar
penalty reward is used here instead, which is required for LLM-guided search.
"""

import numpy as np
from environments.base_reward import BaseReward

FAIL_REWARD          = -100_000.0
INFEASIBLE_PENALTY   = 5_000.0   # penalty per unit of nominal constraint violation (kg)
CONSTRAINT_TOLERANCE = 1e-3      # Table 2 Saves et al.: violations < 10^-3 are accepted


class FuelMassReward(BaseReward):

    def get_prompt_blocks(self):
        return None  # use environment's default prompt blocks

    def evaluate(self, run_sim_fn, design_path: str, case_dir: str) -> tuple:
        result = run_sim_fn(design_path, case_dir)

        if not result.get('success', False):
            return FAIL_REWARD, {
                'metrics': {'fuel_mass': None, 'static_margin': None},
                'images':  result.get('images', []),
                'feedback': f"MDA failed: {result.get('error', '')}",
            }

        fuel_mass     = result['fuel_mass']
        static_margin = result['static_margin']

        if fuel_mass is None or np.isnan(fuel_mass):
            return FAIL_REWARD, {
                'metrics': {'fuel_mass': None, 'static_margin': static_margin},
                'images':  result.get('images', []),
                'feedback': 'NaN fuel mass',
            }

        # Static margin is required to evaluate the constraint; a missing or
        # NaN value means the constraint cannot be assessed — treat as failed.
        if static_margin is None or np.isnan(static_margin):
            return FAIL_REWARD, {
                'metrics': {'fuel_mass': fuel_mass, 'static_margin': None},
                'images':  result.get('images', []),
                'feedback': 'NaN static_margin — constraint unevaluable, treated as failed',
            }

        # Constraint penalty: 0.05 < static_margin < 0.10 (Table 2).
        # Violations within CONSTRAINT_TOLERANCE are accepted per the paper;
        # violation magnitude is measured from the nominal bound, not the
        # tolerance boundary, so the penalty scales correctly with severity.
        penalty = 0.0
        violation_msgs = []

        if static_margin < 0.05 - CONSTRAINT_TOLERANCE:
            viol = 0.05 - static_margin
            penalty += viol * INFEASIBLE_PENALTY
            violation_msgs.append(f'sm={static_margin:.4f} < 0.05 (viol={viol:.4f})')
        elif static_margin > 0.10 + CONSTRAINT_TOLERANCE:
            viol = static_margin - 0.10
            penalty += viol * INFEASIBLE_PENALTY
            violation_msgs.append(f'sm={static_margin:.4f} > 0.10 (viol={viol:.4f})')

        reward = -fuel_mass - penalty

        feedback_parts = [f'fuel={fuel_mass:.1f} kg', f'sm={static_margin:.4f}']
        if violation_msgs:
            feedback_parts.append('INFEASIBLE: ' + ', '.join(violation_msgs))
        feedback = '  |  '.join(feedback_parts)

        return reward, {
            'metrics': {
                'fuel_mass':     fuel_mass,
                'static_margin': static_margin,
                'penalty':       penalty,
            },
            'images':   result.get('images', []),
            'feedback': feedback,
        }
