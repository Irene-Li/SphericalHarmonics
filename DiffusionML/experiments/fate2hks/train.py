"""
Training script: predict HKS from cell fate markers.

Uses DiffusionNet to learn a per-vertex mapping from cell fate marker fields
to Heat Kernel Signatures on organoid surfaces.

Usage:
    python ML/experiments/fate2hks/train.py --data_path Data/20260224 --epochs 200
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

from dataset import CrossPredictionDataset
from model import DiffusionNetPredictor


def area_weighted_mse(pred, target, mass):
    """
    Area-weighted MSE loss (discretisation-invariant).

    Args:
        pred:   (V, C) predicted features
        target: (V, C) target features
        mass:   (V,) area weights
    Returns:
        scalar loss
    """
    diff_sq = (pred - target) ** 2
    weighted = diff_sq * mass.unsqueeze(-1)
    return weighted.sum() / (mass.sum() * pred.shape[-1])


def train_one_epoch(model, dataset, indices, optimizer, device, epoch):
    """Train for one epoch over the given indices."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(indices, desc=f"  Train epoch {epoch}", leave=False)
    for idx in pbar:
        sample = dataset[idx]

        hks = sample['hks'].to(device)
        fate = sample['fate_fields'].to(device)
        mass = sample['mass'].to(device)
        L = sample['L'].to(device)
        evals = sample['evals'].to(device)
        evecs = sample['evecs'].to(device)
        gradX = sample['gradX'].to(device)
        gradY = sample['gradY'].to(device)

        optimizer.zero_grad()
        # Input: fate markers, Target: HKS
        hks_pred = model(fate, mass, L, evals, evecs, gradX, gradY)
        loss = area_weighted_mse(hks_pred, hks, mass)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_samples += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, dataset, indices, device):
    """Evaluate on the given indices."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for idx in indices:
        sample = dataset[idx]

        hks = sample['hks'].to(device)
        fate = sample['fate_fields'].to(device)
        mass = sample['mass'].to(device)
        L = sample['L'].to(device)
        evals = sample['evals'].to(device)
        evecs = sample['evecs'].to(device)
        gradX = sample['gradX'].to(device)
        gradY = sample['gradY'].to(device)

        hks_pred = model(fate, mass, L, evals, evecs, gradX, gradY)
        loss = area_weighted_mse(hks_pred, hks, mass)

        total_loss += loss.item()
        n_samples += 1

    return total_loss / max(n_samples, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train DiffusionNet: cell fate markers -> HKS")
    parser.add_argument('--data_path', type=str, default='Data/small_meshes',
                        help='Path to small_meshes directory')
    parser.add_argument('--output_dir', type=str, default='DiffusionML/experiments/fate2hks/outputs',
                        help='Directory for saved models')
    parser.add_argument('--op_cache_dir', type=str, default='DiffusionML/op_cache',
                        help='Directory to cache DiffusionNet operators')

    # Model hyperparameters
    parser.add_argument('--k_eig', type=int, default=128,
                        help='Number of eigenvalues for DiffusionNet')
    parser.add_argument('--C_width', type=int, default=64,
                        help='Internal DiffusionNet width')
    parser.add_argument('--N_block', type=int, default=4,
                        help='Number of DiffusionNet blocks')
    parser.add_argument('--no_gradient_features', action='store_true',
                        help='Disable gradient features')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (0 = train on all)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Limit dataset to first N samples (0 = use all)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset ---
    print("Loading dataset...")
    dataset = CrossPredictionDataset(
        data_path=args.data_path,
        k_eig=args.k_eig,
        op_cache_dir=args.op_cache_dir,
    )

    if args.max_samples > 0:
        dataset.entries = dataset.entries[:args.max_samples]
        print(f"Truncated to {len(dataset.entries)} samples")

    n_hks = dataset.n_hks
    n_fates = dataset.n_fates

    print(f"\nTask: Fate markers ({n_fates} channels) -> HKS ({n_hks} channels)")
    print(f"  Fate markers: {dataset.fate_names}")

    # --- Training ---
    all_indices = np.arange(len(dataset))

    if args.n_folds > 1:
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
            print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")

            model = DiffusionNetPredictor(
                C_in=n_fates,
                C_out=n_hks,
                C_width=args.C_width,
                N_block=args.N_block,
                mlp_hidden_dims=[args.C_width],
                dropout=False,
                with_gradient_features=not args.no_gradient_features,
                with_gradient_rotations=not args.no_gradient_features,
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_val_loss = float('inf')
            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(train_idx)
                train_loss = train_one_epoch(
                    model, dataset, train_idx, optimizer, device, epoch)

                if epoch % 10 == 0 or epoch == args.epochs:
                    val_loss = evaluate(model, dataset, val_idx, device)
                    print(f"  Epoch {epoch:3d} | train: {train_loss:.6f} | val: {val_loss:.6f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(),
                                   os.path.join(args.output_dir, f'model_fold{fold}.pt'))

            fold_results.append(best_val_loss)
            print(f"  Best val loss: {best_val_loss:.6f}")

        print(f"\nCV results: {np.mean(fold_results):.6f} +/- {np.std(fold_results):.6f}")

    # --- Final model: train on all data ---
    print("\n=== Training final model on all data ===")
    model = DiffusionNetPredictor(
        C_in=n_fates,
        C_out=n_hks,
        C_width=args.C_width,
        N_block=args.N_block,
        mlp_hidden_dims=[args.C_width],
        dropout=False,
        with_gradient_features=not args.no_gradient_features,
        with_gradient_rotations=not args.no_gradient_features,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(all_indices)
        train_loss = train_one_epoch(
            model, dataset, all_indices, optimizer, device, epoch)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d} | loss: {train_loss:.6f}")

    # Save final model
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'model_final.pt'))

    # Save config for later loading
    config = {
        'task': 'fate2hks',
        'C_in': n_fates,
        'C_out': n_hks,
        'C_width': args.C_width,
        'N_block': args.N_block,
        'fate_names': dataset.fate_names,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nModel and config saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
