"""
Credit-Ranked DAG Builder — parameter-free, variable-N.

Builds a DAG by choosing an ordering over active agents, then greedily adds
directed edges from earlier to later agents with bounded in-degree and
out-degree, ensuring DAG acyclicity.

One fixed Aggregator role is appended as the terminal DAG node.
"""

import hashlib
import random
from typing import Dict, List, Tuple, Optional


def _role_stage_rank(role_type: Optional[str]) -> int:
    if role_type == "router":
        return -1
    if role_type == "validator":
        return 1
    return 0


def _seeded_random_order(role_names: List[str], order_seed: int) -> List[str]:
    seed_material = f"{order_seed}::" + "||".join(role_names)
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big", signed=False))
    ordered = list(role_names)
    rng.shuffle(ordered)
    return ordered


def build_credit_dag(
    role_names: List[str],          # active agents (excluding aggregator)
    fast_credits: List[float],      # credit[i] = Fast Credit of role_names[i]
    aggregator_name: str = "Aggregator",
    role_types: Optional[Dict[str, str]] = None,
    b_in: Optional[int] = None,     # max in-degree per node (default: N//2)
    b_out: Optional[int] = None,    # max out-degree per node (default: 2)
    flat: bool = False,             # A1b ablation: deterministic flat topology
    random_order: bool = False,     # A1b ablation: keep DAG construction but randomize node order
    order_seed: int = 0,
) -> List[Tuple[str, str]]:
    """
    Build a directed acyclic graph for the communication topology.

    Returns: list of (source, target) edges, plus edges to the aggregator.
    Final node in topological order is the aggregator.

    If flat=True: return a deterministic linear chain in input order.
    If random_order=True: keep the greedy DAG builder, but choose a deterministic
    pseudo-random node order instead of credit order.
    """
    n = len(role_names)
    if n == 0:
        return []

    if flat:
        # Flat fallback: preserve retrieval order and remove credit-informed edges.
        ordered = list(role_names)
        edges = [(ordered[i], ordered[i + 1]) for i in range(n - 1)]
        for r in ordered:
            edges.append((r, aggregator_name))
        return edges

    if random_order:
        sorted_names = _seeded_random_order(role_names, order_seed)
    else:
        # Keep validators after specialist-like roles so critique/review nodes do not
        # dominate the front of the DAG purely because of their raw fast-credit score.
        order = sorted(
            range(n),
            key=lambda i: (
                _role_stage_rank((role_types or {}).get(role_names[i])),
                -fast_credits[i],
                i,
            ),
        )
        sorted_names = [role_names[i] for i in order]

    # Default degree bounds
    b_out_eff = b_out if b_out is not None else max(1, min(2, n - 1))
    b_in_eff = b_in if b_in is not None else max(1, n // 2)

    # Greedy edge addition: from higher-credit (earlier) to lower-credit (later)
    out_degree = {r: 0 for r in sorted_names}
    in_degree = {r: 0 for r in sorted_names}
    edges: List[Tuple[str, str]] = []

    for i, src in enumerate(sorted_names):
        for j in range(i + 1, n):
            dst = sorted_names[j]
            if out_degree[src] >= b_out_eff:
                break
            if in_degree[dst] >= b_in_eff:
                continue
            edges.append((src, dst))
            out_degree[src] += 1
            in_degree[dst] += 1

    # Every agent sends to the aggregator (terminal node)
    for r in sorted_names:
        edges.append((r, aggregator_name))

    return edges


def topological_order(
    role_names: List[str],
    edges: List[Tuple[str, str]],
    aggregator_name: str = "Aggregator",
) -> List[str]:
    """
    Return roles in topological order (sources before sinks).
    Aggregator is always last.
    """
    from collections import defaultdict, deque

    all_nodes = list(role_names) + [aggregator_name]
    in_deg = defaultdict(int)
    adj = defaultdict(list)

    for src, dst in edges:
        if src in set(all_nodes) and dst in set(all_nodes):
            adj[src].append(dst)
            in_deg[dst] += 1

    queue = deque([n for n in all_nodes if in_deg[n] == 0 and n != aggregator_name])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for nb in adj[node]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0 and nb != aggregator_name:
                queue.append(nb)

    if aggregator_name not in result:
        result.append(aggregator_name)
    return result


def get_incoming_messages(
    role: str,
    edges: List[Tuple[str, str]],
    all_responses: dict,
) -> List[str]:
    """Return list of response strings from agents that have an edge → role."""
    return [all_responses[src] for src, dst in edges if dst == role and src in all_responses]


def compute_dag_levels(
    topo_order: List[str],
    edges: List[Tuple[str, str]],
    aggregator_name: str = "Aggregator",
) -> List[List[str]]:
    """
    Compute parallel execution levels for the DAG (excluding aggregator).

    Nodes at the same level have no inter-dependencies and can run concurrently.
    Level 0: no incoming edges (sources).
    Level k: all predecessors are in levels < k.

    Returns list of levels, each level is a list of role names that can
    run in parallel.
    """
    non_agg = [n for n in topo_order if n != aggregator_name]
    node_set = set(non_agg)

    # Build predecessor map restricted to non-aggregator nodes
    predecessors: dict = {n: set() for n in non_agg}
    for src, dst in edges:
        if src in node_set and dst in node_set:
            predecessors[dst].add(src)

    levels: List[List[str]] = []
    completed: set = set()
    remaining = list(non_agg)  # preserve topo order within each level

    while remaining:
        ready = [n for n in remaining if predecessors[n].issubset(completed)]
        if not ready:
            # Fallback: take first remaining node (shouldn't happen with valid DAG)
            ready = [remaining[0]]
        levels.append(ready)
        for n in ready:
            completed.add(n)
        remaining = [n for n in remaining if n not in completed]

    return levels
