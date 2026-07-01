"""
Learn FULL per-feature HKS-spectrum weights from chamfer + hand-picked groups.

Every (mode, vocab) feature gets its own free weight (an unconstrained
n_modes * n_vocab weight matrix; no factorized modes-by-vocab product). The power
spectrum is cut at MODE_CUT modes so high-frequency surface detail (small
imperfections) is dropped, and selected vocab channels can be deleted (DROP_VOCAB).

Two supervision signals, both acting on the weighted-HKS distance d_w(i,j):
  (A) chamfer correlation: among organoid pairs with chamfer below CONF_THR (small
      chamfer = trustworthy shape similarity), maximize Pearson(d_w, chamfer).
      One-sided — large chamfer is unreliable and excluded. Chamfer is ONLY used
      here, as weak weight-learning supervision; never as ground truth elsewhere.
  (B) group compactness: hand-picked groups (utils.UID_GROUPS, from
      Data/uid_groups.json) — including a 'spheres' group of the lowest-complexity
      organoids — are each pulled TIGHT relative to the overall data spread
      (compactness only, NO between-group repulsion):
          minimize  mean_within_group(d_w) / global_spread(w).
      Positive-unlabelled safe: similar groups are never pushed apart, and
      unlabelled organoids are never treated as negatives.

Objective (minimized):
    -correlation_chamfer  +  beta_group * group_compactness  +  l2 * ||log_weights||^2

Optimizer: mini-batch Adam over the LOG weights, with several random restarts; the
restart with the best objective wins, and the final weights are the across-restart
ensemble mean with unstable (high-CV) features zeroed. Pure numpy.

Saves the learned (n_modes, n_vocab) weight matrix + metadata so dim_red.ipynb can
load and apply it.

Run in the scmpx env:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/anaconda3/envs/scmpx/bin/python optimize_hks_weights.py
"""

import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr

from src import utils

weight_index = 1 


# ---------------------------------------------------------------------------
# Weighted distance and gradients
#   feature weight        w_f          = exp(log_weight_f)
#   squared weight        w_f^2        = exp(2 * log_weight_f)
#   weighted distance     d_w(i, j)    = sqrt( sum_f w_f^2 * (X[i,f] - X[j,f])^2 )
# All gradients are taken wrt the LOG weights (the optimised variable).
# ---------------------------------------------------------------------------

def weighted_distances(pair_sq_diffs, log_weights):
    """Weighted Euclidean distance for each pair: sqrt(sum_f w_f^2 * sq_diff_f)."""
    squared_weights = np.exp(2.0 * log_weights)
    return np.sqrt(pair_sq_diffs @ squared_weights + 1e-14)


def neg_correlation_and_grad(log_weights, pair_sq_diffs, chamfer_z):
    """-Pearson(d_w, chamfer) over the given pairs, and its gradient wrt log_weights.

    `chamfer_z` is the standardised chamfer for these pairs, so the correlation is
    mean(distance_z * chamfer_z). Returns (-correlation, gradient).
    """
    squared_weights = np.exp(2.0 * log_weights)
    distances = np.sqrt(pair_sq_diffs @ squared_weights + 1e-14)
    distances_centered = distances - distances.mean()
    distances_std = np.sqrt((distances_centered * distances_centered).mean()) + 1e-12
    correlation = ((distances_centered / distances_std) * chamfer_z).mean()

    n_pairs = len(distances)
    grad_wrt_distance = (chamfer_z / distances_std
                         - correlation * distances_centered / distances_std ** 2) / n_pairs
    grad_wrt_squared_weights = (grad_wrt_distance / (2.0 * distances)) @ pair_sq_diffs
    grad_wrt_log_weights = grad_wrt_squared_weights * (2.0 * squared_weights)
    return -correlation, -grad_wrt_log_weights


def _mean_distance_and_grad(pair_sq_diffs, squared_weights):
    """Mean weighted distance over the pairs, plus its gradient wrt squared_weights."""
    distances = np.sqrt(pair_sq_diffs @ squared_weights + 1e-14)
    grad_wrt_squared_weights = (pair_sq_diffs / (2.0 * distances)[:, None]).mean(axis=0)
    return distances.mean(), grad_wrt_squared_weights


def group_compactness_and_grad(log_weights, group_pair_sq_diffs, feature_variance):
    """Group-compactness term and its gradient wrt log_weights.

        compactness = mean_within_group(d_w) / global_spread(w)
        global_spread(w) = sqrt( sum_f w_f^2 * feature_variance_f )

    Pulls each group tight relative to the overall spread of the data (no
    between-group repulsion). Returns (compactness, gradient).
    """
    squared_weights = np.exp(2.0 * log_weights)
    mean_within, grad_mean_within = _mean_distance_and_grad(group_pair_sq_diffs, squared_weights)
    global_spread = np.sqrt(feature_variance @ squared_weights + 1e-14)
    grad_global_spread = feature_variance / (2.0 * global_spread)

    compactness = mean_within / global_spread
    grad_wrt_squared_weights = (
        (grad_mean_within * global_spread - mean_within * grad_global_spread)
        / global_spread ** 2)
    grad_wrt_log_weights = grad_wrt_squared_weights * (2.0 * squared_weights)
    return compactness, grad_wrt_log_weights


def combined_objective(log_weights, corr_sq_diffs, chamfer_values,
                       group_sq_diffs, feature_variance, beta_group):
    """(correlation, compactness, score-to-minimize) for a given weight vector."""
    correlation = pearsonr(weighted_distances(corr_sq_diffs, log_weights), chamfer_values)[0]
    if len(group_sq_diffs):
        squared_weights = np.exp(2.0 * log_weights)
        compactness = (np.sqrt(group_sq_diffs @ squared_weights + 1e-14).mean()
                       / np.sqrt(feature_variance @ squared_weights + 1e-14))
    else:
        compactness = 0.0
    score = -correlation + beta_group * compactness
    return correlation, compactness, score


# ---------------------------------------------------------------------------
# Building the supervision pairs
# ---------------------------------------------------------------------------

def confident_pairs(chamfer_path, organoid_features, id_to_index,
                    threshold=None, percentile=None):
    """Small-chamfer ('confident') organoid pairs read from a chamfer npz.

    Selects pairs with chamfer < `threshold` (absolute) or below the given
    `percentile`. Returns:
        pair_sq_diffs   (n_pairs, n_features)  squared feature differences per pair
        chamfer_values  (n_pairs,)             chamfer distance for each pair
        involved_ids    set                    organoid ids appearing in any pair
        cutoff_used     float                  the chamfer cutoff actually applied
        n_organoids     int                    organoids contributing pairs
    Chamfer ids absent from `id_to_index` are dropped with a warning.
    """
    chamfer_npz = np.load(chamfer_path, allow_pickle=True)
    chamfer_ids = chamfer_npz["ids"].astype(str)
    chamfer_matrix = chamfer_npz["C"]

    in_master = np.array([uid in id_to_index for uid in chamfer_ids])
    if not in_master.all():
        print(f"  [warn] {(~in_master).sum()}/{len(chamfer_ids)} ids in {chamfer_path} "
              f"not in master — dropped")
        chamfer_matrix = chamfer_matrix[np.ix_(in_master, in_master)]
        chamfer_ids = chamfer_ids[in_master]

    feature_rows = np.array([id_to_index[uid] for uid in chamfer_ids])
    upper_i, upper_j = np.triu_indices(len(chamfer_ids), k=1)
    chamfer_values = chamfer_matrix[upper_i, upper_j]

    cutoff = threshold if threshold is not None else np.percentile(chamfer_values, percentile)
    keep = chamfer_values < cutoff
    pair_i, pair_j = upper_i[keep], upper_j[keep]

    pair_sq_diffs = (organoid_features[feature_rows[pair_i]]
                     - organoid_features[feature_rows[pair_j]]) ** 2
    involved_ids = set(chamfer_ids[pair_i]) | set(chamfer_ids[pair_j])
    return pair_sq_diffs, chamfer_values[keep], involved_ids, float(cutoff), len(chamfer_ids)


# ---------------------------------------------------------------------------
# Mini-batch Adam with multiple restarts
# ---------------------------------------------------------------------------

def adam_fit(corr_sq_diffs, chamfer_values, group_sq_diffs, feature_variance, beta_group,
             n_restarts=12, epochs=300, batch_size=4096, learning_rate=0.05,
             l2=1e-4, log_weight_clip=4.0, seed=0, verbose=True):
    """Mini-batch Adam over the log weights, with multiple random restarts.

    Maximizes the chamfer correlation while keeping the hand-picked groups compact.
    Returns a dict with the best (lowest-score) restart, plus every restart's
    normalised weights and metrics (used for the stability analysis in main()).
    """
    n_pairs, n_features = corr_sq_diffs.shape
    best = {"score": np.inf, "log_weights": None, "correlation": np.nan, "compactness": np.nan}
    restart_weights, restart_metrics = [], []

    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart)
        # restart 0 starts from the unweighted metric (all weights 1 -> log_weights 0)
        log_weights = np.zeros(n_features) if restart == 0 else rng.normal(0.0, 0.3, n_features)
        adam_m = np.zeros(n_features)
        adam_v = np.zeros(n_features)
        step = 0

        for _ in range(epochs):
            shuffled = rng.permutation(n_pairs)
            for batch_start in range(0, n_pairs, batch_size):
                batch = shuffled[batch_start:batch_start + batch_size]
                batch_chamfer = chamfer_values[batch]
                batch_chamfer_z = (batch_chamfer - batch_chamfer.mean()) / (batch_chamfer.std() + 1e-12)

                _, gradient = neg_correlation_and_grad(log_weights, corr_sq_diffs[batch], batch_chamfer_z)
                if beta_group > 0 and len(group_sq_diffs):
                    _, grad_group = group_compactness_and_grad(log_weights, group_sq_diffs, feature_variance)
                    gradient = gradient + beta_group * grad_group
                gradient = gradient + l2 * log_weights

                step += 1
                adam_m = beta1 * adam_m + (1 - beta1) * gradient
                adam_v = beta2 * adam_v + (1 - beta2) * (gradient * gradient)
                m_hat = adam_m / (1 - beta1 ** step)
                v_hat = adam_v / (1 - beta2 ** step)
                log_weights -= learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)
                np.clip(log_weights, -log_weight_clip, log_weight_clip, out=log_weights)

        correlation, compactness, score = combined_objective(
            log_weights, corr_sq_diffs, chamfer_values, group_sq_diffs, feature_variance, beta_group)
        # normalise so weights from different restarts are comparable (||w|| = sqrt(n_features))
        weights = np.exp(log_weights)
        weights *= np.sqrt(n_features) / np.linalg.norm(weights)
        restart_weights.append(weights)
        restart_metrics.append((correlation, compactness, score))
        if verbose:
            print(f"  restart {restart:2d}: Pearson={correlation:.4f}  "
                  f"group_compactness={compactness:.4f}  score={score:.4f}")
        if score < best["score"]:
            best = {"score": score, "log_weights": log_weights.copy(),
                    "correlation": correlation, "compactness": compactness}

    best["restart_weights"] = np.array(restart_weights)     # (n_restarts, n_features)
    best["restart_metrics"] = np.array(restart_metrics)     # (n_restarts, 3): corr, comp, score
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", default="Data/npz/master.npz")
    parser.add_argument("--chamfer", default="Data/npz/chamfer_update.npz")
    parser.add_argument("--out", default="Data/npz/hks_weights_full.npz")
    parser.add_argument("--mode_cut", type=int, default=8,
                        help="keep only the first this-many HKS modes (drop fine surface detail)")
    parser.add_argument("--drop_vocab", type=int, nargs="*", default=[],
                        help="vocab indices to delete entirely (the drop mask), e.g. --drop_vocab 2 4; "
                             "deleted vocabs get zero weight and the rest adapt to their absence")
    parser.add_argument("--conf_thr", type=float, default=0.9,
                        help="keep organoid pairs with chamfer below this absolute threshold as "
                             "confident correlation supervision (see chamfer distribution plot)")
    parser.add_argument("--beta_group", type=float, default=0.15,
                        help="weight of the hand-picked-group compactness term")
    parser.add_argument("--cv_threshold", type=float, default=0.1,
                        help="zero out features whose weight coefficient-of-variation across "
                             "restarts exceeds this (unstable = poorly determined = unimportant)")
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()

    # ---- load features and apply the vocab-drop mask ------------------------
    organoid_ids, power_spectrum = utils.load_power_spectrum(args.master, mode_cut=args.mode_cut)
    n_modes, n_vocab = power_spectrum.shape[1], power_spectrum.shape[2]
    n_features = n_modes * n_vocab

    drop_vocab = sorted(set(args.drop_vocab))
    if drop_vocab:
        assert 0 <= min(drop_vocab) and max(drop_vocab) < n_vocab, \
            f"drop_vocab {drop_vocab} out of range [0, {n_vocab - 1}]"
        power_spectrum[:, weight_index:, drop_vocab] = 0.0          # deleted vocabs contribute nothing to any distance

    organoid_features = power_spectrum.reshape(len(organoid_ids), -1)    # (N, n_features)
    id_to_index = {uid: i for i, uid in enumerate(organoid_ids)}

    # ---- (A) correlation supervision: high-complexity confident pairs -------
    corr_sq_diffs, chamfer_values, involved_ids, chamfer_cutoff, _ = confident_pairs(
        args.chamfer, organoid_features, id_to_index, threshold=args.conf_thr)
    trusted_ids = np.array(sorted(involved_ids))

    # ---- (B) compactness supervision: hand-picked groups (incl. 'spheres') --
    group_i, group_j, _, _ = utils.group_pair_indices(utils.UID_GROUPS, id_to_index)
    group_sq_diffs = ((organoid_features[group_i] - organoid_features[group_j]) ** 2
                      if len(group_i) else np.zeros((0, n_features)))
    feature_variance = organoid_features.var(axis=0)        # per-feature spread anchor

    print(f"modes kept: {n_modes} (cut at {args.mode_cut}) | vocabs dropped: {drop_vocab} | "
          f"features: {n_features} ({n_modes}x{n_vocab})")
    print(f"correlation: {len(chamfer_values)} high-complexity confident pairs "
          f"(chamfer < {chamfer_cutoff:g})")
    print(f"compactness: {len(group_i)} hand-picked-group pairs (pulled close, no repulsion)")
    unweighted_pearson = pearsonr(np.sqrt(corr_sq_diffs.sum(axis=1)), chamfer_values)[0]
    print(f"unweighted Pearson = {unweighted_pearson:.4f}\nfitting (Adam, {args.restarts} restarts, "
          f"beta_group={args.beta_group})...")

    best = adam_fit(corr_sq_diffs, chamfer_values, group_sq_diffs, feature_variance, args.beta_group,
                    n_restarts=args.restarts, epochs=args.epochs, batch_size=args.batch,
                    learning_rate=args.lr, l2=args.l2)

    # ---- stability across restarts ------------------------------------------
    # A feature whose weight jumps around between restarts is poorly determined by
    # the data (unimportant); a consistently high weight is a real signal.
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    restart_weights = best["restart_weights"]               # (n_restarts, n_features)
    weight_mean = restart_weights.mean(axis=0).reshape(n_modes, n_vocab)
    weight_cv = (restart_weights.std(axis=0)
                 / (np.abs(restart_weights.mean(axis=0)) + 1e-9)).reshape(n_modes, n_vocab)
    print(f"\nbest single restart: Pearson={best['correlation']:.4f}  comp={best['compactness']:.4f}")
    print(f"\n=== stability across {len(restart_weights)} restarts ({n_modes} modes x {n_vocab} vocab) ===")
    print(f"mean weight:\n{np.round(weight_mean, 3)}")
    print(f"CV = std/mean  (high = fluctuates = unimportant):\n{np.round(weight_cv, 2)}")

    # ---- final weights = ENSEMBLE MEAN with unstable (high-CV) features zeroed
    stable_mask = weight_cv <= args.cv_threshold
    if not stable_mask.any():
        print(f"  [warn] every feature has CV > {args.cv_threshold} — restarts disagree "
              f"(unstable objective). Keeping all features (no CV pruning); consider lowering "
              f"--beta_group.")
        stable_mask = np.ones_like(stable_mask)

    weights = np.where(stable_mask, weight_mean, 0.0)
    if drop_vocab:
        weights[weight_index:, drop_vocab] = 0.0                        # deleted vocabs -> zero weight everywhere
    if np.linalg.norm(weights) == 0:
        raise SystemExit(f"all features pruned at cv_threshold={args.cv_threshold} — raise it")
    weights *= np.sqrt(n_features) / np.linalg.norm(weights)    # renormalise the kept weights

    final_distances = np.sqrt(corr_sq_diffs @ (weights.reshape(-1) ** 2) + 1e-14)
    final_pearson = pearsonr(final_distances, chamfer_values)[0]
    final_spearman = spearmanr(final_distances, chamfer_values).correlation
    pruned_features = np.argwhere(~stable_mask)
    print(f"\n=== final weights: ensemble mean, {len(pruned_features)}/{n_features} features zeroed "
          f"(CV > {args.cv_threshold}){f', vocabs {drop_vocab} deleted' if drop_vocab else ''} ===")
    print(f"Pearson={final_pearson:.4f}  Spearman={final_spearman:.4f}   "
          f"(best single restart Pearson was {best['correlation']:.4f})")
    print(f"zeroed (mode, vocab): {[(int(m), int(v)) for m, v in pruned_features]}")
    print(np.round(weights, 3))

    np.savez(args.out, weights=weights, mode_cut=args.mode_cut, conf_thr=args.conf_thr,
             beta_group=args.beta_group, cv_threshold=args.cv_threshold,
             drop_vocab=np.array(drop_vocab, dtype=int),
             pearson=final_pearson, n_pruned=len(pruned_features),
             mean_w=weight_mean, cv_w=weight_cv,
             trusted_ids=trusted_ids, chamfer=args.chamfer,
             restarts_w=restart_weights.reshape(-1, n_modes, n_vocab),  # (n_restarts, n_modes, n_vocab)
             restarts_meta=best["restart_metrics"])                     # (n_restarts, 3): corr, comp, score
    print(f"\nsaved -> {args.out}  (weights {weights.shape}, mode_cut={args.mode_cut}, "
          f"{len(trusted_ids)} trusted ids)")


if __name__ == "__main__":
    main()
