"""
SERO: Self-Evolving Role Orchestration

A credit-guided multi-agent framework that learns to evolve role pools for
decomposable planning tasks. Key contribution: diagnosis and resolution of the
three-level credit-bias paradox that systematically degrades terminal
synthesis/validation roles in credit-guided MAS.

Three-Level Credit Paradox (and fixes):
  L1 Pool Removal     — aggregator gets low RMU → REMOVE action
                        Fix: protected=True on aggregator/validator roles
  L2 Active-Set Excl. — protected roles excluded from top-N retrieval
                        Fix: always include protected roles in active set
  L3 Format Loss      — evolved roles lose anchor communication protocol
                        Fix: format_inherit=True propagates format anchors

Core modules:
  phase_a.py      — Inference: role retrieval, DAG construction, message passing
  trainer.py      — Training: REINFORCE controller, LOO credit, pool evolution
  credit_engine.py — Credit tracking: EMA, fast credit, LOO precise credit
  dag_builder.py  — Credit-ranked DAG topology
  role_card.py    — Role definitions and seed pools (Trip, Calendar, Olympiad, Meeting)
  config.py       — SeroConfig hyperparameters
  executor.py     — Role execution and format inheritance
  controller.py   — ADD/REMOVE/NOOP policy network
"""
