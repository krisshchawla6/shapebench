"""Particle Swarm Optimizer — velocity/position update (Eq. 2-3).

Reference:
    https://doi.org/10.2514/6.2025-3228


Velocity update (Eq. 2):
    v_i^{t+1} = w * v_i^t
              + c1 * r1 * (p_i^t - x_i^t)
              + c2 * r2 * (g^t   - x_i^t)
    x_i^{t+1} = x_i^t + v_i^{t+1}

Linear coefficient schedule (Eq. 3):
    coeff(t) = start - (t/T) * (start - end)

    w:  start=0.8, end=0.2   (inertia)
    c1: start=1.5, end=0.5   (cognitive / personal-best pull)
    c2: start=0.2, end=3.0   (social    / global-best  pull)
"""

import numpy as np

W_START,  W_END  = 0.8, 0.2
C1_START, C1_END = 1.5, 0.5
C2_START, C2_END = 0.2, 3.0


class Swarm:
    """Synchronous PSO swarm over a continuous bounded design space."""

    def __init__(self, n_particles: int, lb: np.ndarray, ub: np.ndarray):
        self.n   = n_particles
        self.dim = len(lb)
        self.lb  = lb.copy()
        self.ub  = ub.copy()
        span     = ub - lb

        self.x = lb + np.random.rand(n_particles, self.dim) * span
        self.v = np.random.uniform(-0.1 * span, 0.1 * span, (n_particles, self.dim))

        self.p     = self.x.copy()
        self.pbest = np.full(n_particles, -np.inf)

        self.g     = self.x[0].copy()
        self.gbest = -np.inf

    def update_bests(self, rewards: np.ndarray):
        """Update personal and global bests from a batch of rewards."""
        for i in range(self.n):
            if rewards[i] > self.pbest[i]:
                self.pbest[i] = rewards[i]
                self.p[i]     = self.x[i].copy()
        best_i = int(np.argmax(rewards))
        if rewards[best_i] > self.gbest:
            self.gbest = rewards[best_i]
            self.g     = self.x[best_i].copy()

    def step(self, t: int, T: int):
        """Apply velocity + position update for all particles at step t of T."""
        frac = t / T
        w  = W_START  - frac * (W_START  - W_END)
        c1 = C1_START - frac * (C1_START - C1_END)
        c2 = C2_START - frac * (C2_START - C2_END)

        r1 = np.random.rand(self.n, self.dim)
        r2 = np.random.rand(self.n, self.dim)

        self.v = (w  * self.v
                  + c1 * r1 * (self.p - self.x)
                  + c2 * r2 * (self.g - self.x))
        self.x = np.clip(self.x + self.v, self.lb, self.ub)
