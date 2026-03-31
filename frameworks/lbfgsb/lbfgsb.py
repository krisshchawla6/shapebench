"""L-BFGS-B optimizer wrapper — thin shim around scipy.optimize.minimize."""

import numpy as np
from scipy.optimize import minimize


class LBFGSOptimizer:
    """Gradient ascent via L-BFGS-B (minimizes -reward).

    Parameters
    ----------
    x0 : np.ndarray
        Initial design vector.
    lb, ub : np.ndarray
        Per-parameter lower and upper bounds.
    maxiter : int
        Maximum L-BFGS-B iterations.
    eps : float
        Finite-difference step size for gradient estimation (3-point).
    """

    def __init__(self, x0, lb, ub, maxiter=200, eps=1e-4):
        self.x0 = x0.copy()
        self.bounds = list(zip(lb.tolist(), ub.tolist()))
        self.maxiter = maxiter
        self.eps = eps

    def run(self, reward_fn, callback=None):
        """Run the optimizer.

        Parameters
        ----------
        reward_fn : callable
            x -> float  (higher is better)
        callback : callable, optional
            Invoked after each L-BFGS-B iteration with the current x.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
        """
        def objective(x):
            return -reward_fn(x)  # L-BFGS-B minimizes

        return minimize(
            objective,
            self.x0,
            method='L-BFGS-B',
            jac='3-point',
            bounds=self.bounds,
            options={
                'maxiter': self.maxiter,
                'eps': self.eps,
                'ftol': 1e-9,
                'gtol': 1e-6,
            },
            callback=callback,
        )
