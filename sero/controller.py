"""
Factorized Contextual-Bandit Controller (~677K params, CPU PyTorch).

Architecture:
  State s_t = [feedback_emb (384) | pool_mean_emb (384) | credit_stats (5)] → 773-dim
  Shared encoder: Linear(773, 256) → ReLU → Linear(256, 256) → ReLU → h_t (256)
  Op head:     Linear(256, |ops|) → softmax
  Target head: for each role r_i: Linear(256+d_op_emb, 256) → Linear(256,1) → scalar
               score_i = MLP([h_t | op_emb | role_emb | c̄_i | recent_φ̂_i])
               + γ / (n_updates_i + 1)

Training: REINFORCE with EMA baseline, batch-normalized reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np

from sero.config import SeroConfig
from sero.credit_engine import CreditEngine
from sero.role_card import RoleCard, is_role_removable


# ── Op-type embedding ─────────────────────────────────────────────────────────

class OpEmbedding(nn.Module):
    """Learnable embedding for operation types."""
    def __init__(self, n_ops: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)

    def forward(self, op_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(op_idx)


# ── Controller ────────────────────────────────────────────────────────────────

class SeroController(nn.Module):
    """
    Factorized contextual-bandit policy for role pool evolution.

    Input state:
        feedback_emb: (d_e,) — sentence embedding of task + answer
        pool_mean_emb: (d_e,) — mean embedding of active roles
        credit_stats: (5,) — [mean_ema, std_ema, min_ema, max_ema, mean_phi]

    Output:
        op_logits: (n_ops,)
        target_scores: (|pool|,) — unnormalized, for selected op_type
    """

    OPS = ["add_anchor", "remove", "noop"]

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: Optional[int] = None) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def __init__(self, config: SeroConfig):
        super().__init__()
        cfg = config
        self.config = config

        d_state = cfg.d_e + cfg.d_e + cfg.credit_dim
        d_h = cfg.d_h
        n_ops = len(self.OPS)
        d_op = cfg.op_emb_dim

        # Shared state encoder: default depth preserves the original 2-layer MLP.
        self.encoder = self._build_mlp(
            input_dim=d_state,
            hidden_dim=d_h,
            hidden_layers=cfg.controller_encoder_layers,
        )

        # Op head
        self.op_head = nn.Linear(d_h, n_ops)

        # Op-type embedding (for target head)
        self.op_emb = OpEmbedding(n_ops, d_op)

        # Role embedding projection: role system_prompt embedding → d_e
        # (role embs come from the external encoder; we project with a linear)
        self.role_proj = nn.Linear(cfg.d_e, d_h)

        # Target head: [h_t | op_emb | role_proj | c̄ | φ̂] → scalar.
        d_target_in = d_h + d_op + d_h + 1 + 1
        self.target_head = self._build_mlp(
            input_dim=d_target_in,
            hidden_dim=d_h,
            hidden_layers=cfg.controller_target_layers,
            output_dim=1,
        )

        self.exploration_gamma = config.exploration_gamma

    @property
    def param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode_state(
        self,
        feedback_emb: torch.Tensor,    # (d_e,)
        pool_mean_emb: torch.Tensor,   # (d_e,)
        credit_stats: torch.Tensor,    # (5,)
    ) -> torch.Tensor:
        """Encode state to shared hidden vector h_t (d_h,)."""
        s = torch.cat([feedback_emb, pool_mean_emb, credit_stats], dim=-1)  # (773,)
        return self.encoder(s)                                                # (d_h,)

    def op_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Op-type logits (n_ops,)."""
        return self.op_head(h)

    def target_scores(
        self,
        h: torch.Tensor,               # (d_h,)
        op_idx: int,
        role_embs: torch.Tensor,       # (|pool|, d_e)
        ema_credits: torch.Tensor,     # (|pool|, 1)
        recent_phis: torch.Tensor,     # (|pool|, 1)
        n_updates: torch.Tensor,       # (|pool|,) int tensor
    ) -> torch.Tensor:
        """
        Target scores for each role, given op type.
        Returns (|pool|,) tensor of unnormalized scores.
        """
        n_roles = role_embs.size(0)
        op_emb = self.op_emb(torch.tensor(op_idx, dtype=torch.long))  # (d_op,)

        # Project role embeddings
        rp = self.role_proj(role_embs)  # (n, d_h)

        # Expand h and op_emb to (n, ...)
        h_exp = h.unsqueeze(0).expand(n_roles, -1)       # (n, d_h)
        op_exp = op_emb.unsqueeze(0).expand(n_roles, -1)  # (n, d_op)

        feat = torch.cat([h_exp, op_exp, rp, ema_credits, recent_phis], dim=-1)  # (n, 578)
        raw_scores = self.target_head(feat).squeeze(-1)  # (n,)

        # Exploration bonus: γ / (n_updates + 1)
        bonus = self.exploration_gamma / (n_updates.float() + 1.0)
        return raw_scores + bonus

    def forward(
        self,
        feedback_emb: torch.Tensor,
        pool_mean_emb: torch.Tensor,
        credit_stats: torch.Tensor,
        role_embs: torch.Tensor,
        ema_credits: torch.Tensor,
        recent_phis: torch.Tensor,
        n_updates: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Full forward pass.
        Returns:
            op_logits: (n_ops,)
            target_scores_per_op: list of (n_roles,) tensors, one per op type
        """
        h = self.encode_state(feedback_emb, pool_mean_emb, credit_stats)
        op_log = self.op_logits(h)
        target_per_op = [
            self.target_scores(h, i, role_embs, ema_credits, recent_phis, n_updates)
            for i in range(len(self.OPS))
        ]
        return op_log, target_per_op


# ── Action sampling ───────────────────────────────────────────────────────────

def build_op_action_mask(
    pool: List[RoleCard],
    config: SeroConfig,
    allow_remove: bool = True,
    required_capability_families: Optional[set[str]] = None,
) -> torch.Tensor:
    """Build an op-level action mask from current pool constraints.

    The controller should not sample actions that cannot be executed under the
    current pool size / protection constraints.
    """
    ops = SeroController.OPS
    mask = torch.ones(len(ops), dtype=torch.bool)

    add_idx = ops.index("add_anchor")
    remove_idx = ops.index("remove")

    if not pool or len(pool) >= config.p_max:
        mask[add_idx] = False

    required_role_type_minima = None
    if config.conditional_validator_pass and config.validator_slots > 0:
        required_role_type_minima = {"validator": config.validator_slots}

    removable_exists = any(
        is_role_removable(
            pool,
            role.name,
            protect_critical_roles=config.protect_critical_roles,
            required_capability_families=required_capability_families,
            required_role_type_minima=required_role_type_minima,
        )
        for role in pool
    )
    if (not allow_remove) or len(pool) <= config.p_min or not removable_exists:
        mask[remove_idx] = False

    return mask

def sample_action(
    op_logits: torch.Tensor,
    target_scores_per_op: List[torch.Tensor],
    pool: List[RoleCard],
    use_credit_state: bool = True,
    deterministic: bool = False,
    config: Optional[SeroConfig] = None,
    required_capability_families: Optional[set[str]] = None,
) -> Tuple[str, Optional[str], torch.Tensor]:
    """
    Sample (op_type, target_role_name, log_prob_tensor) from the controller output.

    Returns log_prob as a Tensor with grad_fn so REINFORCE can backprop through it.

    For NOOP: target = None
    For ADD-anchor / REMOVE: target = role name from pool

    Returns:
        op_type: str ("add_anchor" | "remove" | "noop")
        target: Optional[str] — role name or None
        log_prob: torch.Tensor (scalar, retains grad) — joint log probability
    """
    ops = SeroController.OPS
    op_probs = F.softmax(op_logits, dim=-1)

    if deterministic:
        op_idx = int(op_probs.argmax().item())
    else:
        op_idx = int(torch.multinomial(op_probs, 1).item())

    log_prob_op = torch.log(op_probs[op_idx] + 1e-9)
    op_type = ops[op_idx]

    if op_type == "noop":
        return op_type, None, log_prob_op

    if not pool:
        invalid_op = "invalid_add" if op_type == "add_anchor" else "invalid_remove"
        return invalid_op, None, log_prob_op

    # Target selection
    t_scores = target_scores_per_op[op_idx]  # (|pool|,)

    # Mask protected roles for REMOVE to avoid phantom actions
    if op_type == "remove":
        protect_critical_roles = True if config is None else config.protect_critical_roles
        mask = torch.tensor([
            not is_role_removable(
                pool,
                role.name,
                protect_critical_roles=protect_critical_roles,
                required_capability_families=required_capability_families,
                required_role_type_minima=(
                    {"validator": config.validator_slots}
                    if config is not None and config.conditional_validator_pass and config.validator_slots > 0
                    else None
                ),
            )
            for role in pool
        ], dtype=torch.bool)
        if mask.all():
            return "invalid_remove", None, log_prob_op
        t_scores = t_scores.clone()
        t_scores[mask] = float("-inf")
        if not torch.isfinite(t_scores).any():
            return "invalid_remove", None, log_prob_op

    t_probs = F.softmax(t_scores, dim=-1)

    if deterministic:
        t_idx = int(t_probs.argmax().item())
    else:
        t_idx = int(torch.multinomial(t_probs, 1).item())

    log_prob_target = torch.log(t_probs[t_idx] + 1e-9)
    target_name = pool[t_idx].name

    return op_type, target_name, log_prob_op + log_prob_target


# ── State construction helpers ────────────────────────────────────────────────

def build_controller_tensors(
    pool: List[RoleCard],
    credit_engine: CreditEngine,
    role_embs: np.ndarray,             # (|pool|, d_e) — pre-computed
    feedback_emb: np.ndarray,          # (d_e,)
    pool_mean_emb: np.ndarray,         # (d_e,)
    use_credit_state: bool = True,
):
    """
    Convert numpy arrays and credit engine state into torch tensors
    ready for the SeroController forward pass.
    """
    if use_credit_state:
        credit_stats = credit_engine.credit_stats([r.name for r in pool])
        ema_credits = np.array([[credit_engine.get_ema(r.name)] for r in pool], dtype=np.float32)
        recent_phis = np.array([[credit_engine.get_recent_phi(r.name)] for r in pool], dtype=np.float32)
        n_upd = np.array([credit_engine.get_n_updates(r.name) for r in pool], dtype=np.int64)
    else:
        # A1a ablation: remove all controller-visible credit memory features.
        credit_stats = np.zeros(5, dtype=np.float32)
        ema_credits = np.zeros((len(pool), 1), dtype=np.float32)
        recent_phis = np.zeros((len(pool), 1), dtype=np.float32)
        n_upd = np.zeros(len(pool), dtype=np.int64)

    return {
        "feedback_emb": torch.tensor(feedback_emb, dtype=torch.float32),
        "pool_mean_emb": torch.tensor(pool_mean_emb, dtype=torch.float32),
        "credit_stats": torch.tensor(credit_stats, dtype=torch.float32),
        "role_embs": torch.tensor(role_embs, dtype=torch.float32),
        "ema_credits": torch.tensor(ema_credits, dtype=torch.float32),
        "recent_phis": torch.tensor(recent_phis, dtype=torch.float32),
        "n_updates": torch.tensor(n_upd, dtype=torch.long),
    }


def sample_random_action(
    pool: List[RoleCard],
    op_mask: torch.Tensor,
    config: Optional[SeroConfig] = None,
    required_capability_families: Optional[set[str]] = None,
) -> Tuple[str, Optional[str], torch.Tensor]:
    """Uniformly sample a legal controller action without using policy logits."""
    ops = SeroController.OPS
    valid_op_indices = [idx for idx, allowed in enumerate(op_mask.tolist()) if allowed]
    if not valid_op_indices:
        return "noop", None, torch.tensor(0.0, dtype=torch.float32)

    op_idx = int(np.random.choice(valid_op_indices))
    op_type = ops[op_idx]

    if op_type == "noop":
        return op_type, None, torch.tensor(0.0, dtype=torch.float32)

    if op_type == "add_anchor":
        return op_type, None, torch.tensor(0.0, dtype=torch.float32)

    protect_critical_roles = True if config is None else config.protect_critical_roles
    required_role_type_minima = None
    if config is not None and config.conditional_validator_pass and config.validator_slots > 0:
        required_role_type_minima = {"validator": config.validator_slots}

    removable_roles = [
        role.name
        for role in pool
        if is_role_removable(
            pool,
            role.name,
            protect_critical_roles=protect_critical_roles,
            required_capability_families=required_capability_families,
            required_role_type_minima=required_role_type_minima,
        )
    ]
    if not removable_roles:
        return "invalid_remove", None, torch.tensor(0.0, dtype=torch.float32)

    target_name = str(np.random.choice(removable_roles))
    return op_type, target_name, torch.tensor(0.0, dtype=torch.float32)
