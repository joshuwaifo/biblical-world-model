"""
Test-time value function — the only thing that updates during a session.

Alberta Plan design (arxiv 2208.11173, Sutton, Bowling, Pilarski):
  - Temporally uniform: updated on every time step, not in special training phases
  - Predicts expected cumulative alignment reward from any Biblical coordinate
  - Updated via TD(λ): eligibility traces give credit across the full trajectory

The BibleGAT weights (176 params) are FROZEN.  This module holds 9 learnable
parameters (8 weights + 1 bias) — a linear readout of the 8-dim Biblical state.
That is intentional: the manifold already encodes structure; the value function
only needs to learn "which directions in this session lead to high alignment."

Architecture (Alberta Plan Fig 2):
  Perception  →  state s_t (8-dim, maintained externally)
  Value Fn    →  V(s_t)   = w·s_t + b            [updated here]
  Policy      →  π(a|s_t) = oracle.next_action    [frozen GNN attention]
  Transition  →  T(s,a)   = expected next state   [frozen encoder + EMA]
"""

import numpy as np


class ValueFunction:
    """
    Linear TD(λ) value function over 8-dim Biblical coordinate space.

    Implements the Alberta Plan's value-function component:
    - Learns from the alignment reward signal on every time step
    - Uses eligibility traces for multi-step credit assignment
    - Runs in the foreground (temporally uniform, no offline phases)
    """

    def __init__(
        self,
        dim: int = 8,
        gamma: float = 0.95,   # discount — values future alignment
        eta: float = 0.05,     # learning rate
        lmbda: float = 0.8,    # eligibility trace decay
    ):
        self.dim    = dim
        self.gamma  = gamma
        self.eta    = eta
        self.lmbda  = lmbda

        # The 9 learnable parameters (NOT GNN weights)
        self.w: np.ndarray = np.zeros(dim, dtype=np.float32)
        self.b: float      = 0.0

        # Eligibility trace — carries credit backward across steps
        self.e: np.ndarray = np.zeros(dim, dtype=np.float32)

        # Diagnostics
        self.n_updates: int   = 0
        self.td_errors: list  = []

    def predict(self, s: np.ndarray) -> float:
        """V(s) = w·s + b"""
        return float(np.dot(self.w, s.astype(np.float32)) + self.b)

    def update(self, s_t: np.ndarray, r_t: float,
               s_t1: np.ndarray, done: bool = False) -> float:
        """
        TD(λ) update — called on every navigation step.

        δ_t  = r_t + γ·V(s_{t+1}) - V(s_t)     [TD error]
        e_t  = γ·λ·e_{t-1} + s_t                [eligibility trace]
        w   += η·δ_t·e_t
        b   += η·δ_t

        Returns the TD error δ_t (used for diagnostics).
        """
        s_t  = s_t.astype(np.float32)
        s_t1 = s_t1.astype(np.float32)

        v_t  = self.predict(s_t)
        v_t1 = 0.0 if done else self.predict(s_t1)
        delta = r_t + self.gamma * v_t1 - v_t

        # Accumulate eligibility trace
        self.e = self.gamma * self.lmbda * self.e + s_t

        # Parameter update
        self.w += self.eta * delta * self.e
        self.b += self.eta * delta

        self.n_updates += 1
        self.td_errors.append(delta)
        return delta

    def reset_trace(self) -> None:
        """Reset eligibility trace at start of each new query session."""
        self.e[:] = 0.0

    def state_dict(self) -> dict:
        return {
            "w":         self.w.copy(),
            "b":         self.b,
            "n_updates": self.n_updates,
            "mean_td_error": float(np.mean(self.td_errors[-20:])) if self.td_errors else 0.0,
        }
