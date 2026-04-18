"""L-BFGS-B optimizer wrapper — thin shim around scipy.optimize.minimize."""

import numpy as np
from scipy.optimize import minimize


class LBFGSOptimizer:
    """Gradient ascent via L-BFGS-B (minimizes -reward).

    Internally normalizes all parameters to [0, 1] so that the FD step size
    ``eps`` is scale-invariant. Without normalization, a fixed ``eps`` is
    negligibly small for parameters with large ranges (e.g. [100, 700]),
    producing near-zero gradient estimates and premature convergence.

    Normalization is applied unconditionally rather than as an opt-in flag.
    For environments with small parameter ranges (NeuralFoil: ~[-0.5, 0.6],
    DrivAer: ~[-0.1, 0.2]) the effect is negligible (~10% change in FD step
    size). For environments with large ranges (BlendedNet: B3=[200, 700],
    alpha=[-3, 3]) it is essential — without it, eps=1e-4 perturbs B3 by
    only 2e-7 of its range, the gradient is numerically zero, and L-BFGS-B
    converges after ~27 evals at the starting point. This was the root cause
    of the failed BlendedNet lbfgsb benchmark runs (April 2026).
    Note: NeuralFoil and DrivAer_Star lbfgsb benchmark runs were completed
    before this normalization was introduced and were not rerun. For those
    environments the old fixed-eps behaviour was adequate (parameter ranges
    ~0.2–1.1), so the published benchmark results are unaffected.

    Parameters
    ----------
    x0 : np.ndarray
        Initial design vector (original parameter space).
    lb, ub : np.ndarray
        Per-parameter lower and upper bounds (original parameter space).
    maxiter : int
        Maximum L-BFGS-B iterations.
    eps : float
        Finite-difference step size in normalized [0, 1] space.
        1e-4 gives a 0.01% perturbation per parameter regardless of scale.
    """

    def __init__(self, x0, lb, ub, maxiter=200, eps=1e-4):
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.scale = self.ub - self.lb
        self.x0_norm = (x0 - self.lb) / self.scale
        self.maxiter = maxiter
        self.eps = eps

    def _denorm(self, x_norm):
        return self.lb + x_norm * self.scale

    def run(self, reward_fn, callback=None):
        """Run the optimizer.

        Parameters
        ----------
        reward_fn : callable
            x -> float  (higher is better, original parameter space)
        callback : callable, optional
            Invoked after each L-BFGS-B iteration with x in original space.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
        """
        def objective(x_norm):
            return -reward_fn(self._denorm(x_norm))

        def callback_wrapper(x_norm):
            if callback is not None:
                callback(self._denorm(x_norm))

        bounds_norm = [(0.0, 1.0)] * len(self.x0_norm)

        return minimize(
            objective,
            self.x0_norm,
            method='L-BFGS-B',
            jac='3-point',
            bounds=bounds_norm,
            options={
                'maxiter': self.maxiter,
                'eps': self.eps,
                'ftol': 1e-9,
                'gtol': 1e-6,
            },
            callback=callback_wrapper,
        )
