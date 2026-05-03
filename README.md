# Certified Robustness for Temporal GNNs via Shared-Noise Smoothing

A tighter temporal certificate for Graph Neural Networks via shared randomness — sound under the standard Neyman–Pearson construction, strictly tighter than per-window Bonferroni, and validated on synthetic fraud, Cora-temporal, and real Elliptic Bitcoin.

> **CS 763: Trustworthy AI · Phase 3** · University of Wisconsin–Madison <br>
> **Authors:** Rohan Gupta (`rkgupta3@wisc.edu`) · Jeevesh Mahajan (`jmahajan2@wisc.edu`)

---

## Overview

Existing certified-robustness work for GNNs targets a single static snapshot. Real fraud graphs evolve over time, and the natural extension — running one Neyman–Pearson certificate per window and union-bounding across them — pays a Bonferroni penalty of α/T on every window. We replace that with **one** edge-flip realization shared across all T windows, sum the per-window softmax outputs, and apply a single Clopper–Pearson bound at the **full** target rate α. Sound, strictly tighter, and dataset-agnostic.

## Headline result

Certified accuracy at R = 2 (the median Nettack budget):

| Dataset                  | Edge homophily | Aggregate | Per-window (Bonf.) | **Shared-noise (ours)** | Δ vs. aggregate |
|--------------------------|----------------|-----------|--------------------|-------------------------|-----------------|
| Synthetic fraud (N=1000) | 0.44           | 0.383     | 0.133              | **0.500**               | +0.117          |
| Synthetic fraud (N=2000) | 0.44           | 0.561     | 0.233              | **0.722**               | +0.161          |
| Cora-temporal            | 0.81           | 0.238     | 0.095              | **0.405**               | +0.167          |
| Real Elliptic Bitcoin    | 0.95           | 0.267     | 0.383              | **0.683**               | **+0.417**      |

The shared-noise certificate strictly beats both alternatives in every experiment. The largest gain — **+41.7 percentage points** — is on Elliptic, the only experiment whose temporal dynamics are real rather than synthesized.

## Contributions

1. **Shared-noise temporal certificate.** A new temporal-composition construction in which a single global edge-flip realization is sampled, T per-window forwards are run on the same realized graph, the per-window softmax outputs are summed, and a single Clopper–Pearson lower bound is applied at full α. Sound under the standard Neyman–Pearson argument; strictly tighter than the per-window Bonferroni union bound.
2. **Heterophily-aware Bernoulli smoothing** (Phase 2). A non-uniform smoothing distribution that weights edge-flip probabilities by neighborhood similarity, preserving the cross-class signal that fraud detection relies on.
3. **Cross-spectrum empirical validation.** Identical certificate construction evaluated across edge homophily 0.44 to 0.95 and 2.7k to 204k nodes.
4. **Tightening sanity check.** Re-running synthetic fraud at N=2000 widens the headline R=2 gap from +0.117 to +0.161, confirming the improvement is structural rather than a small-sample Clopper–Pearson artefact.

## Setup

Core dependencies: `torch`, `torch-geometric`, `numpy`, `scipy`, `statsmodels` (for Clopper–Pearson), `pandas`, `matplotlib`.

### Datasets

- **Synthetic fraud** is generated on the fly by `src/data/synthetic_fraud.py`.
- **Cora-temporal** uses Planetoid's standard split with each node hashed into one of 24 buckets to synthesize timesteps.
- **Elliptic Bitcoin** must be downloaded separately from the [Kaggle release by Weber et al. (2019)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and placed under `data/elliptic/`.

## Reproducing the results

Each experiment trains its own uniform-noise GCN once, then runs all three certificate variants on the same trained model.

```bash
# Phase 1 baselines (GCN / GAT / GraphSAGE on synthetic fraud)
python experiments/phase1_baselines.py

# Phase 3 certificate variants
python experiments/exp1_fraud_n1000.py
python experiments/exp2_fraud_n2000.py
python experiments/exp3_cora.py
python experiments/exp4_elliptic.py
```

Default smoothing hyperparameters (held constant across variants):

| Hyperparameter | Value              | Notes                                            |
|----------------|--------------------|--------------------------------------------------|
| `p_smooth`     | 0.10               | Per-edge flip probability                        |
| `p_add`        | 0.01               | Bernoulli random addition                        |
| `N0`           | 100                | Calibration samples (top-class selection)        |
| `N`            | 1000 / 2000 / 1500 | Certification samples (fraud / fraud-rerun / Elliptic) |
| `α`            | 0.01               | Target failure rate (Bonferroni: `α/T`)          |
| `T`            | 3                  | Temporal window count                            |

## Method overview

```
                      ξ ~ Π   (one global edge-flip mask)
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   G_w1 ∪ ξ|w1    G_w2 ∪ ξ|w2    G_w3 ∪ ξ|w3
        │               │               │
   softmax f(·)    softmax f(·)    softmax f(·)
        └───────────────┼───────────────┘
                        ▼
              Σ over w  →  ĉ = argmax s(v)
                        ▼
       Single Clopper–Pearson bound at full α
                        ▼
              Certified radius R*  (no Bonferroni)
```

**Why it is sound.** The certificate is a deterministic function of *one* randomness source ξ, so the standard Neyman–Pearson argument applies to the aggregated softmax-sum statistic exactly as to a single forward pass.

**Why it is tighter.** Per-window outputs from one trained model on overlapping graph regions are positively correlated. The Bonferroni union bound is tight only under independence; shared-noise exploits the actual correlation, strictly dominating Bonferroni whenever it is positive.

## Limitations and future work

- **Certified radius ceiling.** All four experiments yield R* = 0 at R ≥ 4. Sweeping `p_smooth` and reporting accuracy–radius Pareto curves is a natural next step.
- **Window count is fixed.** All experiments use T = 3; the shared-noise advantage should grow with T because the Bonferroni penalty grows in T.
- **Adaptive attacker evaluation deferred.** Adversaries that target the realized mask ξ rather than the static graph remain to be benchmarked.
- **Heuristic defense baselines.** GNN-Guard and Jaccard-pruning were left out to keep the focus on certificate variants.


## Key references

- Mujkanovic et al. *Are Defenses for Graph Neural Networks Robust?* NeurIPS 2022.
- Bojchevski, Klicpera & Günnemann. *Efficient Robustness Certificates for Discrete Data.* ICML 2020.
- Cohen, Rosenfeld & Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML 2019.
- Weber et al. *Anti-Money Laundering in Bitcoin.* KDD-AML 2019.
- Zügner, Akbarnejad & Günnemann. *Adversarial Attacks on Neural Networks for Graph Data (Nettack).* KDD 2018.
- Pareja et al. *EvolveGCN.* AAAI 2020.

## Acknowledgements

Thanks to **Prof. Somesh Jha** and the CS 763 staff for guidance across all three project phases.
