"""
Credit Engine — Role Marginal Utility at three granularities.

Fast Credit:    per-round heuristic, drives DAG topology ordering
Precise Credit: Shapley LOO (leave-one-out), updates EMA
Historical Credit: EMA across episodes, drives retrieval + controller state
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np


# ── Fast Credit ────────────────────────────────────────────────────────────────

def fast_credit(
    response_emb: np.ndarray,    # (d_e,)
    query_emb: np.ndarray,       # (d_e,)
    consensus_emb: np.ndarray,   # (d_e,)
    alpha: float = 0.7,
) -> float:
    """
    Fast Credit c_i^k for one agent in one round.
    c_i^k = alpha * sim(resp, query) + (1-alpha) * sim(resp, consensus)

    Higher = more query-relevant and more consistent with consensus.
    """
    sim_q = float(_cosine(response_emb, query_emb))
    sim_c = float(_cosine(response_emb, consensus_emb))
    return alpha * sim_q + (1.0 - alpha) * sim_c


def compute_round_fast_credits(
    response_embs: List[np.ndarray],   # list of (d_e,) — one per active agent
    query_emb: np.ndarray,
    alpha: float = 0.7,
) -> List[float]:
    """
    Compute Fast Credits for all active agents in one round.
    Consensus = mean of all response embeddings.
    """
    if not response_embs:
        return []
    consensus = np.mean(response_embs, axis=0)
    return [fast_credit(e, query_emb, consensus, alpha) for e in response_embs]


# ── Precise Credit (Shapley LOO) ───────────────────────────────────────────────

def loo_precise_credit(
    score_with: float,
    score_without: float,
) -> float:
    """
    Precise Credit: phi_hat_i = s(P_t) - s(P_t \ {r_i})
    Positive = role helps; negative = role hurts.
    """
    return score_with - score_without


# ── Historical Credit (EMA) ────────────────────────────────────────────────────

@dataclass
class RoleCredit:
    """Per-role credit state tracked by the CreditEngine."""
    role_name: str
    ema_credit: float = 0.0          # c_bar_{r_i}
    recent_phi: float = 0.0          # most recent LOO estimate
    n_updates: int = 0               # times this role has been LOO-evaluated
    last_fast_credit: float = 0.0    # most recent Fast Credit (per-round)


class CreditEngine:
    """
    Stateful credit tracker for the role pool.

    Usage:
        ce = CreditEngine(mu=0.1, alpha=0.7)
        ce.register_role("Job Prioritizer", parent_ema=None)
        ce.update_ema("Job Prioritizer", phi_hat=0.3)
        stats = ce.credit_stats()  # 5-dim vector for controller state
    """

    def __init__(self, mu: float = 0.1, alpha: float = 0.7):
        self.mu = mu        # EMA decay
        self.alpha = alpha  # Fast Credit query-sim weight
        self._credits: Dict[str, RoleCredit] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register_role(
        self,
        name: str,
        parent_ema: Optional[float] = None,
        init_n_updates: int = 0,
        init_recent_phi: float = 0.0,
    ) -> None:
        """Add a new role with explicit cold-start state.

        parent_ema is kept for backwards compatibility, but callers that want a
        true cold start should pass parent_ema=None and an explicit
        init_n_updates.
        """
        if name not in self._credits:
            init_ema = parent_ema if parent_ema is not None else 0.0
            self._credits[name] = RoleCredit(
                role_name=name,
                ema_credit=init_ema,
                recent_phi=init_recent_phi,
                n_updates=init_n_updates,
            )

    def remove_role(self, name: str) -> None:
        """Remove a role from tracking."""
        self._credits.pop(name, None)

    def sync_with_pool(
        self,
        role_names: List[str],
        parent_ema_map: Optional[Dict[str, float]] = None,
        init_n_updates: int = 0,
    ) -> None:
        """
        Sync credit state to match the current role pool.
        New roles get parent EMA if available; removed roles are dropped.
        """
        current = set(self._credits.keys())
        new = set(role_names)
        for name in current - new:
            self.remove_role(name)
        for name in new - current:
            parent_ema = (parent_ema_map or {}).get(name)
            self.register_role(name, parent_ema=parent_ema, init_n_updates=init_n_updates)

    # ── Updates ───────────────────────────────────────────────────────────────

    def update_ema(self, name: str, phi_hat: float) -> None:
        """Update EMA credit for a role after a LOO evaluation."""
        if name not in self._credits:
            self.register_role(name)
        rc = self._credits[name]
        rc.ema_credit = (1.0 - self.mu) * rc.ema_credit + self.mu * phi_hat
        rc.recent_phi = phi_hat
        rc.n_updates += 1

    def update_fast_credit(self, name: str, fast_c: float) -> None:
        if name in self._credits:
            self._credits[name].last_fast_credit = fast_c

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_ema(self, name: str) -> float:
        return self._credits[name].ema_credit if name in self._credits else 0.0

    def get_recent_phi(self, name: str) -> float:
        return self._credits[name].recent_phi if name in self._credits else 0.0

    def get_n_updates(self, name: str) -> int:
        return self._credits[name].n_updates if name in self._credits else 0

    def get_fast_credit(self, name: str) -> float:
        return self._credits[name].last_fast_credit if name in self._credits else 0.0

    def ema_ranking(self, role_names: List[str]) -> List[str]:
        """Return role names sorted by EMA credit, descending (best first)."""
        return sorted(role_names, key=lambda n: self.get_ema(n), reverse=True)

    def fast_credit_ranking(self, role_names: List[str]) -> List[str]:
        """Return role names sorted by last Fast Credit, descending."""
        return sorted(role_names, key=lambda n: self.get_fast_credit(n), reverse=True)

    def credit_stats(self, role_names: Optional[List[str]] = None) -> np.ndarray:
        """
        5-dim credit statistics vector for controller state:
        [mean(c_bar), std(c_bar), min(c_bar), max(c_bar), mean(recent_phi)]
        """
        names = role_names or list(self._credits.keys())
        if not names:
            return np.zeros(5, dtype=np.float32)
        emas = np.array([self.get_ema(n) for n in names], dtype=np.float32)
        phis = np.array([self.get_recent_phi(n) for n in names], dtype=np.float32)
        return np.array([
            float(emas.mean()),
            float(emas.std()) if len(emas) > 1 else 0.0,
            float(emas.min()),
            float(emas.max()),
            float(phis.mean()),
        ], dtype=np.float32)

    def all_role_info(self, role_names: Optional[List[str]] = None) -> Dict[str, dict]:
        names = role_names or list(self._credits.keys())
        return {
            n: {
                "ema": self.get_ema(n),
                "recent_phi": self.get_recent_phi(n),
                "n_updates": self.get_n_updates(n),
                "fast_credit": self.get_fast_credit(n),
            }
            for n in names
        }

    # ── Serialization ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialize full credit engine state for checkpointing."""
        return {
            "mu": self.mu,
            "alpha": self.alpha,
            "credits": {
                name: {
                    "ema_credit": rc.ema_credit,
                    "recent_phi": rc.recent_phi,
                    "n_updates": rc.n_updates,
                    "last_fast_credit": rc.last_fast_credit,
                }
                for name, rc in self._credits.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore credit engine state from a checkpoint dict."""
        self.mu = state.get("mu", self.mu)
        self.alpha = state.get("alpha", self.alpha)
        self._credits.clear()
        for name, rc_dict in state.get("credits", {}).items():
            self._credits[name] = RoleCredit(
                role_name=name,
                ema_credit=rc_dict.get("ema_credit", 0.0),
                recent_phi=rc_dict.get("recent_phi", 0.0),
                n_updates=rc_dict.get("n_updates", 0),
                last_fast_credit=rc_dict.get("last_fast_credit", 0.0),
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
