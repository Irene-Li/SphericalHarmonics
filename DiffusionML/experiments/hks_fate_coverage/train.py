"""
Training script: DiffusionNet encoder → cell-fate binary classification.

Goal: verify that DiffusionNet can extract useful organoid-level features from
HKS under a simple, well-defined classification objective.  If training succeeds
here but DiffusionNet neurons still die in the VAE, the problem is the
Chamfer + KL objective (or HKS normalisation interacting with the decoder).
If it fails here too, the encoder architecture itself needs rethinking.

Task: for each organoid and each requested fate, predict whether that fate is
present anywhere on the organoid surface (yes/no).

Loss: BCEWithLogitsLoss — binary cross-entropy applied to raw logits.
BCEWithLogitsLoss is preferred over sigmoid + BCELoss for numerical stability.

Each organoid contributes a (n_fates,) binary target:
    label_i = 1  if any vertex has fate_i signal > 0
    label_i = 0  otherwise

No target normalisation is required (labels are already {0, 1}).

Usage:
    python DiffusionML/experiments/hks_fate_coverage/train.py \\
        --data_path Data/small_meshes \\
        --fate_names cycd agr lgr \\
        --epochs 300

Outputs saved to --output_dir:
    model_final.pt         final model weights
    config.json            training configuration
    predictions.npz        per-organoid predicted labels vs true labels
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dataset import FateCoverageDataset
from model import FateCoverageNet


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(logits, target, loss_fn):
    """
    logits: (1, n_fates) raw logits from the model
    target: (n_fates,)   binary ground-truth labels {0, 1}
    loss_fn:  BCEWithLogitsLoss instance
    Returns scalar.
    """
    return loss_fn(logits.squeeze(0), target)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _to_device(t, device):
    """Move tensor to device, densifying sparse tensors for MPS (which has no
    sparse backend). Dense sparse matrices have negligible overhead for the
    mesh sizes used here."""
    if device.type == 'mps' and t.is_sparse:
        return t.to_dense().to(device)
    return t.to(device)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataset, indices, optimizer, device, epoch,
                    loss_fn, grad_clip):
    """
    grad_clip: float, max gradient norm (0 = disabled).
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(indices, desc=f"  Epoch {epoch}", leave=False)
    for idx in pbar:
        sample = dataset[idx]
        hks      = _to_device(sample['hks'],      device)
        coverage = _to_device(sample['coverage'],  device)
        mass     = _to_device(sample['mass'],      device)
        L        = _to_device(sample['L'],         device)
        evals    = _to_device(sample['evals'],     device)
        evecs    = _to_device(sample['evecs'],     device)
        gradX    = _to_device(sample['gradX'],     device)
        gradY    = _to_device(sample['gradY'],     device)

        optimizer.zero_grad()
        logits = model(hks, mass, L, evals, evecs, gradX, gradY)
        loss = compute_loss(logits, coverage, loss_fn)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_samples  += 1
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, dataset, indices, device, loss_fn):
    """
    Returns BCE loss, predicted binary labels, true labels, and organoid IDs.
    Predictions are thresholded at sigmoid(logit) > 0.5.
    """
    model.eval()
    total_loss = 0.0
    n_samples  = 0
    preds_all, targets_all, ids_all = [], [], []

    for idx in indices:
        sample   = dataset[idx]
        hks      = _to_device(sample['hks'],      device)
        coverage = _to_device(sample['coverage'],  device)
        mass     = _to_device(sample['mass'],      device)
        L        = _to_device(sample['L'],         device)
        evals    = _to_device(sample['evals'],     device)
        evecs    = _to_device(sample['evecs'],     device)
        gradX    = _to_device(sample['gradX'],     device)
        gradY    = _to_device(sample['gradY'],     device)

        logits = model(hks, mass, L, evals, evecs, gradX, gradY)
        loss = compute_loss(logits, coverage, loss_fn)

        # Threshold at 0.5 in probability space
        pred_labels = (torch.sigmoid(logits.squeeze(0)) > 0.5).float()

        total_loss  += loss.item()
        n_samples   += 1
        preds_all.append(pred_labels.cpu().numpy())
        targets_all.append(coverage.cpu().numpy())
        ids_all.append(sample['meta']['id'])

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, np.array(preds_all), np.array(targets_all), ids_all


def _make_model(args, n_hks, device):
    return FateCoverageNet(
        C_in=n_hks,
        n_fates=len(args.fate_names),
        C_width=args.C_width,
        N_block=args.N_block,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    ).to(device)


def _per_fate_metrics(preds, targets, fate_names):
    """
    Compute per-fate accuracy and F1 score from binary arrays.

    preds:   (N, n_fates) binary predicted labels
    targets: (N, n_fates) binary ground-truth labels
    Returns list of (accuracy, f1) tuples, one per fate.
    """
    results = []
    for i in range(len(fate_names)):
        p = preds[:, i]
        t = targets[:, i]
        acc = (p == t).mean()
        tp  = ((p == 1) & (t == 1)).sum()
        fp  = ((p == 1) & (t == 0)).sum()
        fn  = ((p == 0) & (t == 1)).sum()
        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)
        results.append((float(acc), float(f1)))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train DiffusionNet for fate presence/absence classification")
    parser.add_argument('--data_path',    type=str, default='Data/small_meshes')
    parser.add_argument('--output_dir',   type=str,
                        default='DiffusionML/experiments/hks_fate_coverage/outputs')
    parser.add_argument('--op_cache_dir', type=str, default='DiffusionML/op_cache')

    # Fate targets
    parser.add_argument('--fate_names', nargs='+', default=['lgr', 'sero', 'lyz'],
                        help='Fate names to predict. Must match field names in the npz data files.')

    # Encoder
    parser.add_argument('--k_eig',     type=int,   default=128)
    parser.add_argument('--C_width',   type=int,   default=64)
    parser.add_argument('--N_block',   type=int,   default=4)
    parser.add_argument('--mlp_hidden',type=int,   default=64,
                        help='Hidden dim in prediction head (0 = linear)')
    parser.add_argument('--dropout',   action='store_true',
                        help='Enable dropout inside DiffusionNet MiniMLPs')

    # Training
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--lr',           type=float, default=1e-4,
                        help='Initial learning rate (default 1e-4; DiffusionNet is sensitive to lr)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip',    type=float, default=1.0,
                        help='Gradient clipping max norm (0 = disabled)')
    parser.add_argument('--lr_min',       type=float, default=None,
                        help='Minimum LR for cosine scheduler (default: lr * 0.01)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'plateau'],
                        help='LR scheduler: none, cosine annealing, or reduce-on-plateau')
    parser.add_argument('--n_folds',      type=int,   default=5)
    parser.add_argument('--max_samples',  type=int,   default=0)
    parser.add_argument('--seed',         type=int,   default=42)

    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Predicting presence/absence of: {args.fate_names}")
    print(f"lr={args.lr}  grad_clip={args.grad_clip}  scheduler={args.lr_scheduler}")

    # Loss function: BCE on raw logits (numerically stable)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Loading dataset...")
    dataset = FateCoverageDataset(
        data_path=args.data_path,
        k_eig=args.k_eig,
        op_cache_dir=args.op_cache_dir,
        fate_names=args.fate_names,
    )
    if args.max_samples > 0:
        dataset.entries = dataset.entries[:args.max_samples]
        print(f"Truncated to {len(dataset.entries)} samples")

    # Print label statistics
    all_labels = []
    for idx in range(len(dataset)):
        npz = np.load(dataset.entries[idx]['data_path'], allow_pickle=True)
        scalars = npz['scalars']
        names   = npz['names'].tolist()
        labs = []
        for field in dataset.fate_names:
            if field in names:
                fidx = names.index(field)
                labs.append(float((scalars[:, fidx] > 0).any()))
            else:
                labs.append(0.0)
        all_labels.append(labs)
    all_labels = np.array(all_labels)
    print(f"\nLabel statistics across {len(dataset)} organoids:")
    print(f"  {'Fate':<10} {'Present':>9} {'Absent':>9} {'%Present':>10}")
    for i, fn in enumerate(args.fate_names):
        n_pos = int(all_labels[:, i].sum())
        n_neg = len(dataset) - n_pos
        pct   = 100.0 * n_pos / len(dataset)
        print(f"  {fn:<10} {n_pos:>9d} {n_neg:>9d} {pct:>9.1f}%")
    print()

    n_hks = dataset.n_hks
    model = _make_model(args, n_hks, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"  Encoder: DiffusionNet C_width={args.C_width} N_block={args.N_block}")
    print(f"  Head:    MLP hidden={args.mlp_hidden} → {len(args.fate_names)} logits")
    print(f"  Loss:    BCEWithLogitsLoss (binary classification)")

    # Naive baseline: always predict the majority class per fate
    naive_acc = np.maximum(all_labels.mean(axis=0),
                           1.0 - all_labels.mean(axis=0)).mean()
    print(f"\nNaive baseline accuracy (majority class): {naive_acc:.4f}")

    all_indices = np.arange(len(dataset))

    def _make_scheduler(optimizer, n_epochs):
        if args.lr_scheduler == 'cosine':
            lr_min = args.lr_min if args.lr_min is not None else args.lr * 0.01
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=lr_min)
        elif args.lr_scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        return None

    if args.n_folds > 1:
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
            print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
            model = _make_model(args, n_hks, device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
            scheduler = _make_scheduler(optimizer, args.epochs)

            best_val_loss = float('inf')
            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(train_idx)
                tr_loss = train_one_epoch(model, dataset, train_idx,
                                          optimizer, device, epoch, loss_fn,
                                          args.grad_clip)

                if epoch % 10 == 0 or epoch == args.epochs:
                    val_loss, val_preds, val_targets, _ = evaluate(
                        model, dataset, val_idx, device, loss_fn)
                    current_lr = optimizer.param_groups[0]['lr']
                    # Per-fate accuracy on val set
                    metrics = _per_fate_metrics(val_preds, val_targets, args.fate_names)
                    mean_acc = np.mean([m[0] for m in metrics])
                    print(f"  Epoch {epoch:3d} | "
                          f"train {tr_loss:.5f} | val {val_loss:.5f} | "
                          f"val_acc {mean_acc:.4f} | lr {current_lr:.2e}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(),
                                   os.path.join(args.output_dir,
                                                f'model_fold{fold}.pt'))
                    if args.lr_scheduler == 'plateau':
                        scheduler.step(val_loss)

                if scheduler is not None and args.lr_scheduler == 'cosine':
                    scheduler.step()

            fold_results.append(best_val_loss)
            print(f"  Best val BCE loss: {best_val_loss:.5f}")

        print(f"\nCV BCE loss: "
              f"{np.mean(fold_results):.5f} ± {np.std(fold_results):.5f}")

    print("\n=== Training final model on all data ===")
    model = _make_model(args, n_hks, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = _make_scheduler(optimizer, args.epochs)

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(all_indices)
        tr_loss = train_one_epoch(model, dataset, all_indices,
                                  optimizer, device, epoch, loss_fn,
                                  args.grad_clip)
        if epoch % 10 == 0 or epoch == args.epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | loss={tr_loss:.5f} | lr={current_lr:.2e}")

        if scheduler is not None and args.lr_scheduler == 'cosine':
            scheduler.step()

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'model_final.pt'))

    config = vars(args)
    config['n_hks'] = int(n_hks)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("\nSaving predictions on full dataset...")
    final_loss, preds, targets, ids = evaluate(
        model, dataset, all_indices, device, loss_fn)
    print(f"Final BCE loss on full dataset: {final_loss:.5f}")

    # Per-fate metrics
    print(f"\nPer-fate metrics (binary classification):")
    print(f"  {'Fate':<10} {'Accuracy':>10} {'F1':>8}")
    metrics = _per_fate_metrics(preds, targets, args.fate_names)
    for fn, (acc, f1) in zip(args.fate_names, metrics):
        print(f"  {fn:<10} {acc:>10.4f} {f1:>8.4f}")

    np.savez(
        os.path.join(args.output_dir, 'predictions.npz'),
        preds=preds,
        targets=targets,
        ids=np.array(ids),
        fate_names=np.array(args.fate_names),
    )
    print(f"\nOutput: {args.output_dir}")


if __name__ == '__main__':
    main()
