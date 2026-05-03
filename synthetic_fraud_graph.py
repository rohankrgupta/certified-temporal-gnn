"""
Synthetic Heterophilous Financial Fraud Graph Generator
=======================================================
Generates a temporal, heterophilous graph mimicking financial fraud patterns.

Output: a PyG Data object with the EXACT same interface as EllipticBitcoinDataset:
  - data.x:          [N, num_features] node feature matrix
  - data.edge_index:  [2, E] edge list
  - data.y:          [N] labels (0=licit, 1=illicit, -1=unknown)
  - data.train_mask: [N] bool
  - data.test_mask:  [N] bool
  - data.t:          [N] integer timestep per node (same role as TIMESTEP)

Drop-in replacement: swap the Elliptic loading cell with:
    from synthetic_fraud_graph import generate_fraud_graph
    data, TIMESTEP = generate_fraud_graph()
Then everything downstream (models, training, evaluation, attacks) works unchanged.

Authors: Rohan Gupta, Jeevesh Mahajan (CS 763)
"""

import numpy as np
import torch
from torch_geometric.data import Data


def generate_fraud_graph(
    # --- Graph sizing ---
    num_timesteps: int = 25,
    nodes_per_timestep: int = 480,       # ~12,000 total nodes
    illicit_ratio: float = 0.15,          # 15% illicit (higher than real for meaningful heterophily)
    unknown_ratio: float = 0.08,          # 8% unlabeled

    # --- Edge structure ---
    avg_degree: float = 10.0,
    target_edge_homophily: float = 0.40,  # target: genuinely heterophilous

    # --- Fraud patterns (fraction of illicit nodes using each) ---
    fan_out_frac: float = 0.30,
    chain_frac: float = 0.40,
    cycle_frac: float = 0.30,

    # --- Feature design ---
    num_features: int = 165,              # match Elliptic's 165 features
    feature_noise_std: float = 0.3,

    # --- Temporal split ---
    train_end_timestep: int = 15,         # timesteps 0..14 = train
    val_end_timestep: int = 18,           # timesteps 15..17 = val
    # timesteps 18..24 = test

    # --- Reproducibility ---
    seed: int = 42,
):
    """Generate a synthetic heterophilous fraud graph.

    Key design: we directly control edge homophily by specifying what fraction
    of edges should be cross-class vs same-class. The fraud patterns (fan-out,
    chain, cycle) determine the STRUCTURE of cross-class edges, while
    target_edge_homophily determines the RATIO.

    Returns:
        data: PyG Data object (same interface as EllipticBitcoinDataset)
        timestep: LongTensor of per-node timesteps
    """
    rng = np.random.default_rng(seed)

    # ================================================================
    # 1. Generate nodes with timestep assignments and labels
    # ================================================================
    total_nodes = num_timesteps * nodes_per_timestep
    timesteps = np.repeat(np.arange(num_timesteps), nodes_per_timestep)

    labels = np.zeros(total_nodes, dtype=np.int64)
    n_illicit = int(total_nodes * illicit_ratio)
    n_unknown = int(total_nodes * unknown_ratio)

    # Distribute illicit nodes across timesteps (more in later timesteps)
    illicit_weights = np.linspace(0.5, 1.5, num_timesteps)
    illicit_weights /= illicit_weights.sum()
    illicit_per_ts = rng.multinomial(n_illicit, illicit_weights)

    illicit_indices = []
    for t in range(num_timesteps):
        ts_nodes = np.where(timesteps == t)[0]
        n_ill_t = min(illicit_per_ts[t], len(ts_nodes) - 1)
        chosen = rng.choice(ts_nodes, size=n_ill_t, replace=False)
        illicit_indices.extend(chosen.tolist())
    illicit_indices = np.array(illicit_indices[:n_illicit])
    labels[illicit_indices] = 1

    # Mark some nodes as unknown
    licit_indices_all = np.where(labels == 0)[0]
    unknown_chosen = rng.choice(licit_indices_all, size=min(n_unknown, len(licit_indices_all)),
                                replace=False)
    labels[unknown_chosen] = -1

    illicit_set = set(illicit_indices.tolist())
    licit_set = set(np.where(labels == 0)[0].tolist())

    # ================================================================
    # 2. Generate edges with CONTROLLED heterophily
    # ================================================================
    target_edges = int(total_nodes * avg_degree / 2)

    # Compute how many cross-class vs same-class edges we need
    n_cross_class = int(target_edges * (1.0 - target_edge_homophily))
    n_same_class = target_edges - n_cross_class

    edge_set = set()

    def add_edge(a, b):
        if a == b:
            return
        key = (min(int(a), int(b)), max(int(a), int(b)))
        edge_set.add(key)

    # Helper: get nodes by label and timestep
    def get_nodes(label, ts_start, ts_end):
        mask = (timesteps >= ts_start) & (timesteps < ts_end)
        if label is not None:
            mask &= (labels == label)
        return np.where(mask)[0]

    # --- 2a. Cross-class edges (heterophilous structure via fraud patterns) ---
    cross_added = 0
    attempts = 0
    max_attempts = n_cross_class * 5

    # First: structured fraud pattern edges
    for t in range(num_timesteps):
        ill_t = get_nodes(1, t, t + 1)
        lic_nearby = get_nodes(0, max(0, t - 2), min(num_timesteps, t + 3))
        ill_nearby = get_nodes(1, max(0, t - 2), min(num_timesteps, t + 3))

        if len(ill_t) == 0 or len(lic_nearby) == 0:
            continue

        n_fan = int(len(ill_t) * fan_out_frac)
        n_chain = int(len(ill_t) * chain_frac)
        n_cycle = int(len(ill_t) * cycle_frac)

        # Fan-out: hub → multiple licit nodes
        fan_hubs = rng.choice(ill_t, size=min(n_fan, len(ill_t)), replace=False)
        for hub in fan_hubs:
            n_spokes = rng.integers(3, 8)
            targets = rng.choice(lic_nearby, size=min(n_spokes, len(lic_nearby)), replace=False)
            for tgt in targets:
                add_edge(hub, tgt)
                cross_added += 1

        # Chain: illicit → illicit → ... → licit
        chain_starts = rng.choice(ill_t, size=min(n_chain, len(ill_t)), replace=False)
        for start in chain_starts:
            chain_len = rng.integers(2, 5)
            current = start
            for step in range(chain_len - 1):
                candidates = ill_nearby[ill_nearby != current]
                if len(candidates) > 0:
                    nxt = rng.choice(candidates)
                    add_edge(current, nxt)  # illicit-illicit (same-class)
                    current = nxt
            # Cash-out to licit
            if len(lic_nearby) > 0:
                add_edge(current, rng.choice(lic_nearby))
                cross_added += 1

        # Cycle: ring of illicit + exit to licit
        if len(ill_t) >= 3:
            n_cycles = max(1, n_cycle // 4)
            for _ in range(n_cycles):
                cycle_size = min(rng.integers(3, 6), len(ill_t))
                ring = rng.choice(ill_t, size=cycle_size, replace=False)
                for j in range(len(ring)):
                    add_edge(ring[j], ring[(j + 1) % len(ring)])
                if len(lic_nearby) > 0:
                    add_edge(ring[0], rng.choice(lic_nearby))
                    cross_added += 1

    # Fill remaining cross-class edges with random illicit-licit connections
    illicit_arr = np.array(list(illicit_set))
    licit_arr = np.array(list(licit_set))
    while cross_added < n_cross_class and attempts < max_attempts:
        a = rng.choice(illicit_arr)
        b = rng.choice(licit_arr)
        # Prefer temporal locality
        if abs(int(timesteps[a]) - int(timesteps[b])) <= 3:
            before = len(edge_set)
            add_edge(a, b)
            if len(edge_set) > before:
                cross_added += 1
        attempts += 1

    # --- 2b. Same-class edges ---
    same_added = 0
    attempts = 0

    # Licit-licit edges (the majority of same-class)
    n_licit_licit = int(n_same_class * 0.85)
    while same_added < n_licit_licit and attempts < n_licit_licit * 3:
        t = rng.integers(0, num_timesteps)
        lic_t = get_nodes(0, t, t + 1)
        lic_nearby = get_nodes(0, max(0, t - 1), min(num_timesteps, t + 2))
        if len(lic_t) > 1 and len(lic_nearby) > 1:
            a = rng.choice(lic_t)
            b = rng.choice(lic_nearby)
            before = len(edge_set)
            add_edge(a, b)
            if len(edge_set) > before:
                same_added += 1
        attempts += 1

    # Illicit-illicit edges (smaller fraction of same-class)
    n_ill_ill = n_same_class - n_licit_licit
    attempts = 0
    while same_added < n_same_class and attempts < n_ill_ill * 5:
        t = rng.integers(0, num_timesteps)
        ill_nearby = get_nodes(1, max(0, t - 2), min(num_timesteps, t + 3))
        if len(ill_nearby) > 1:
            a, b = rng.choice(ill_nearby, size=2, replace=False)
            before = len(edge_set)
            add_edge(a, b)
            if len(edge_set) > before:
                same_added += 1
        attempts += 1

    # --- 2c. Ensure minimum degree for all nodes ---
    for node in range(total_nodes):
        t = timesteps[node]
        node_edges = sum(1 for (a, b) in edge_set if a == node or b == node)
        if node_edges < 2:
            nearby = get_nodes(None, max(0, t - 1), min(num_timesteps, t + 2))
            candidates = nearby[nearby != node]
            if len(candidates) > 0:
                for tgt in rng.choice(candidates, size=min(3, len(candidates)), replace=False):
                    add_edge(node, tgt)

    # Convert to edge_index (undirected = both directions)
    edges = list(edge_set)
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # ================================================================
    # 3. Generate node features
    # ================================================================
    features = rng.standard_normal((total_nodes, num_features)).astype(np.float32)

    # Licit profile
    licit_mask = labels == 0
    features[licit_mask, :10] += 0.5
    features[licit_mask, 10:20] += 0.3
    features[licit_mask, 40:50] += 0.2

    # Illicit profile: distinct but not trivially separable
    illicit_mask = labels == 1
    features[illicit_mask, :10] -= 0.4
    features[illicit_mask, 10:20] -= 0.6
    features[illicit_mask, 20:30] += 1.0
    features[illicit_mask, 30:40] += 0.5
    features[illicit_mask, 50:60] -= 0.3

    # Unknown nodes: noisy blend
    unknown_mask = labels == -1
    features[unknown_mask, :20] += rng.uniform(-0.3, 0.3, (unknown_mask.sum(), 20))

    # Timestep-dependent drift
    for t in range(num_timesteps):
        ts_mask = timesteps == t
        features[ts_mask, 1] += 0.02 * t
        features[ts_mask, 2] -= 0.01 * t

    # Add noise
    features += rng.standard_normal(features.shape).astype(np.float32) * feature_noise_std

    # Encode timestep as feature column 0 (matching Elliptic convention)
    features[:, 0] = timesteps.astype(np.float32)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    t_tensor = torch.tensor(timesteps, dtype=torch.long)

    # ================================================================
    # 4. Build train/val/test masks (temporal split)
    # ================================================================
    labeled = labels >= 0
    train_mask = torch.tensor(labeled & (timesteps < train_end_timestep), dtype=torch.bool)
    val_mask = torch.tensor(
        labeled & (timesteps >= train_end_timestep) & (timesteps < val_end_timestep),
        dtype=torch.bool,
    )
    test_mask = torch.tensor(labeled & (timesteps >= val_end_timestep), dtype=torch.bool)

    # ================================================================
    # 5. Assemble PyG Data object
    # ================================================================
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    data.val_mask = val_mask
    data.t = t_tensor

    return data, t_tensor


def print_graph_stats(data, timestep):
    """Print summary statistics matching what the notebook expects."""
    y = data.y.numpy()
    ei = data.edge_index.numpy()
    ts = timestep.numpy()

    src, dst = ei[0], ei[1]
    labeled = (y >= 0)
    valid = labeled[src] & labeled[dst]
    same_label = y[src[valid]] == y[dst[valid]]
    edge_homo = same_label.mean()

    # Per-class homophily
    ill_nodes = set(np.where(y == 1)[0])
    ill_edges_mask = np.array([int(s) in ill_nodes or int(d) in ill_nodes
                               for s, d in zip(src, dst)])
    ill_valid = valid & ill_edges_mask
    if ill_valid.sum() > 0:
        ill_homo = (y[src[ill_valid]] == y[dst[ill_valid]]).mean()
    else:
        ill_homo = float('nan')

    lic_nodes = set(np.where(y == 0)[0])
    lic_edges_mask = np.array([int(s) in lic_nodes and int(d) in lic_nodes
                               for s, d in zip(src, dst)])
    lic_valid = valid & lic_edges_mask
    if lic_valid.sum() > 0:
        lic_homo = 1.0  # licit-licit edges are all same-class by definition
    else:
        lic_homo = float('nan')

    print(f"=== Synthetic Fraud Graph Statistics ===")
    print(f"Nodes:     {data.num_nodes}")
    print(f"Edges:     {data.num_edges} (undirected: {data.num_edges // 2})")
    print(f"Features:  {data.num_features}")
    print(f"Timesteps: {int(ts.max()) + 1} (range {int(ts.min())}-{int(ts.max())})")
    print(f"")
    print(f"Labels:    licit={int((y==0).sum())}, illicit={int((y==1).sum())}, "
          f"unknown={int((y==-1).sum())}")
    print(f"Illicit %: {(y==1).sum() / (y>=0).sum() * 100:.1f}%")
    print(f"")
    print(f"Train:     {int(data.train_mask.sum())} "
          f"(illicit={int(((y==1) & data.train_mask.numpy()).sum())})")
    if hasattr(data, 'val_mask'):
        print(f"Val:       {int(data.val_mask.sum())} "
              f"(illicit={int(((y==1) & data.val_mask.numpy()).sum())})")
    print(f"Test:      {int(data.test_mask.sum())} "
          f"(illicit={int(((y==1) & data.test_mask.numpy()).sum())})")
    print(f"")
    print(f"Avg degree:     {data.num_edges / data.num_nodes:.1f}")
    print(f"Edge homophily: {edge_homo:.4f}  (target: <0.50)")
    print(f"  Illicit-node edge homophily: {ill_homo:.4f}")
    print(f"  (Cora ~0.81, Citeseer ~0.74)")

    # Per-timestep stats
    print(f"\nPer-timestep breakdown:")
    for t in range(int(ts.max()) + 1):
        mask_t = ts == t
        n_t = mask_t.sum()
        ill_t = ((y == 1) & mask_t).sum()
        # Edges where both endpoints are in this timestep
        e_mask = mask_t[src] & mask_t[dst]
        n_edges_t = e_mask.sum()
        if (valid & e_mask).sum() > 0:
            homo_t = (y[src[valid & e_mask]] == y[dst[valid & e_mask]]).mean()
        else:
            homo_t = float('nan')
        print(f"  t={t:2d}: {n_t:5d} nodes, {ill_t:3d} illicit, "
              f"{n_edges_t:5d} edges, homo={homo_t:.3f}")


if __name__ == "__main__":
    data, ts = generate_fraud_graph()
    print_graph_stats(data, ts)
