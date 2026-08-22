"""
SERO Configuration
All hyperparameters, API settings, and constants in one place.
Set OPENROUTER_API_KEY (and optionally OPENROUTER_BASE_URL) via environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# ── API Config ─────────────────────────────────────────────────────────────────
# Set your API key via the OPENROUTER_API_KEY environment variable.
# The base URL defaults to the official OpenRouter endpoint.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Models via OpenRouter (see https://openrouter.ai/models)
AGENT_MODEL = os.getenv("SERO_AGENT_MODEL", "gpt-4o-mini")   # frozen task agents
EXECUTOR_MODEL = os.getenv("SERO_EXECUTOR_MODEL", "gpt-4o-mini")  # role card editor
# For a cheaper option: "google/gemini-flash-1.5" or "meta-llama/llama-3.1-8b-instruct:free"

# ── Controller ─────────────────────────────────────────────────────────────────
CONTROLLER_D_E = 512        # sentence-transformer embedding dim (jina-embeddings-v2-small-en)
CONTROLLER_D_H = 256        # controller hidden dim
CONTROLLER_CREDIT_DIM = 5   # [mean, std, min, max, mean_phi]
OP_EMB_DIM = 64             # op-type embedding dim
CONTROLLER_ENCODER_LAYERS = 2  # state encoder depth (default preserves current 2-layer MLP)
CONTROLLER_TARGET_LAYERS = 1   # hidden depth in target head before scalar output

# ── Role Pool ──────────────────────────────────────────────────────────────────
P_MAX = 20                  # max role pool size
N_MAX = 5                   # max active agents per task (keep small for API cost)
T_ROUND = 2                 # collaboration rounds per task

# ── Credit ─────────────────────────────────────────────────────────────────────
EMA_MU = 0.1                # EMA decay for historical credit
FAST_CREDIT_ALPHA = 0.5     # Fast Credit: sim weight (1-alpha = consistency weight)

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 8              # tasks per gradient step (small for API cost)
WARMUP_EPOCHS = 3           # train Δscore-only (no credit state)
MAIN_EPOCHS = 7             # main training with full credit
LOO_REFRESH_INTERVAL = 5    # full-pool LOO refresh every N episodes
LOO_MIN_POOL_SIZE = 4       # minimum pool size before LOO credit refresh
NEW_ROLE_INITIAL_N_UPDATES = 3  # conservative cold-start count for new roles
SHUFFLE_TRAIN_TASKS = True  # default legacy behavior; naturalplan runner preserves split order
ENTROPY_BETA = 0.05         # entropy regularization coefficient (higher = more exploration)
EXPLORATION_GAMMA = 0.1     # credit-maintenance exploration bonus weight
LR = 1e-3                   # REINFORCE learning rate
EMA_BASELINE_DECAY = 0.9    # reward baseline EMA decay
NOOP_COLLAPSE_THRESHOLD = 0.85  # early stop if NOOP rate exceeds this

# ── Role selection / validation topology ─────────────────────────────────────
SPECIALIST_SLOTS = 3        # number of non-validator specialists before terminal aggregation
CONDITIONAL_VALIDATOR_PASS = True  # reserve validators outside the main specialist budget
VALIDATOR_SLOTS = 1         # reserved validator roles when conditional validator pass is enabled

# ── Role-pool diversity controls ─────────────────────────────────────────────
PRESERVE_SEED_FAMILY_COVERAGE = True
MIN_CAPABILITY_FAMILY_DIVERSITY = 0.5
MAX_CAPABILITY_FAMILY_DOMINANCE = 0.6
MAX_ROLE_PROMPT_SIMILARITY = 0.92
ORTHOGONAL_ROLE_BLACKLIST_TOP_K = 2

# ── Reward ─────────────────────────────────────────────────────────────────────
REWARD_EPS = 1e-6           # reward normalization epsilon

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
ENCODER_MODEL = "jinaai/jina-embeddings-v2-small-en"  # 8192 tokens, 512d, 33M params

# ── Seed ops ───────────────────────────────────────────────────────────────────
SEED_OPS = ["add_anchor", "remove", "noop"]  # base operation types

@dataclass
class SeroConfig:
    """Full SERO configuration, override fields as needed."""
    # API
    openrouter_api_key: str = field(default_factory=lambda: OPENROUTER_API_KEY)
    openrouter_base_url: str = OPENROUTER_BASE_URL
    agent_model: str = AGENT_MODEL
    executor_model: str = EXECUTOR_MODEL

    # Architecture
    d_e: int = CONTROLLER_D_E
    d_h: int = CONTROLLER_D_H
    credit_dim: int = CONTROLLER_CREDIT_DIM
    op_emb_dim: int = OP_EMB_DIM
    controller_encoder_layers: int = CONTROLLER_ENCODER_LAYERS
    controller_target_layers: int = CONTROLLER_TARGET_LAYERS
    p_max: int = P_MAX
    n_max: int = N_MAX
    t_round: int = T_ROUND

    # Credit
    ema_mu: float = EMA_MU
    fast_credit_alpha: float = FAST_CREDIT_ALPHA

    # Training
    batch_size: int = BATCH_SIZE
    warmup_epochs: int = WARMUP_EPOCHS
    main_epochs: int = MAIN_EPOCHS
    loo_refresh_interval: int = LOO_REFRESH_INTERVAL
    loo_min_pool_size: int = LOO_MIN_POOL_SIZE
    new_role_initial_n_updates: int = NEW_ROLE_INITIAL_N_UPDATES
    shuffle_train_tasks: bool = SHUFFLE_TRAIN_TASKS
    entropy_beta: float = ENTROPY_BETA
    exploration_gamma: float = EXPLORATION_GAMMA
    lr: float = LR
    ema_baseline_decay: float = EMA_BASELINE_DECAY
    noop_collapse_threshold: float = NOOP_COLLAPSE_THRESHOLD
    reward_eps: float = REWARD_EPS

    # Paths
    results_dir: str = RESULTS_DIR
    encoder_model: str = ENCODER_MODEL

    # Experiment controls
    seed: int = 42
    use_credit_state: bool = True    # controller sees credit-memory features during decision-making
    use_active_set_credit: bool = True  # active-role retrieval mixes query similarity with credit ranking
    use_credit_dag: bool = True      # execution topology uses the credit-informed DAG
    random_controller: bool = False  # replace learned controller with uniform random legal actions
    evolve_roles: bool = True        # static-seed baseline: set False
    protect_critical_roles: bool = True  # L1 fix: set False to ablate (allow REMOVE on protected roles)
    freeze_role_pool: bool = False   # joint evolution ablation: sample actions but keep the committed role pool frozen
    post_aggregator_validator_check: bool = True  # run the validator pass after aggregator synthesis
    controller_reward_training: bool = True  # compute REINFORCE loss / backward for the controller
    format_inherit: bool = True      # ablation: set False to disable format-inheritance in executor
    specialist_slots: int = SPECIALIST_SLOTS
    conditional_validator_pass: bool = CONDITIONAL_VALIDATOR_PASS
    validator_slots: int = VALIDATOR_SLOTS
    preserve_seed_family_coverage: bool = PRESERVE_SEED_FAMILY_COVERAGE
    min_capability_family_diversity: float = MIN_CAPABILITY_FAMILY_DIVERSITY
    max_capability_family_dominance: float = MAX_CAPABILITY_FAMILY_DOMINANCE
    max_role_prompt_similarity: float = MAX_ROLE_PROMPT_SIMILARITY
    orthogonal_role_blacklist_top_k: int = ORTHOGONAL_ROLE_BLACKLIST_TOP_K

    # Pool size bounds
    p_min: int = 2                   # minimum pool size (prevent collapse to 1 role)

    def __post_init__(self):
        if self.d_h < 1:
            raise ValueError("d_h must be >= 1")
        if self.controller_encoder_layers < 1:
            raise ValueError("controller_encoder_layers must be >= 1")
        if self.controller_target_layers < 1:
            raise ValueError("controller_target_layers must be >= 1")
        if not self.openrouter_api_key:
            import warnings
            warnings.warn(
                "OPENROUTER_API_KEY not set. Set env var OPENROUTER_API_KEY "
                "or pass openrouter_api_key=<key> to SeroConfig.",
                stacklevel=2
            )
