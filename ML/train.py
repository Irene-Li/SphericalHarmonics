"""
Training script for the DiffusionNet autoencoder on organoid shapes.

Usage:
    python ML/train.py --data_path Data/20260224 --epochs 200

After training, the latent embeddings are saved to ML/outputs/ and can
be loaded in ML/diffusionnet_clustering.ipynb for visualization.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset import OrganoidDataset, collate_single
from model import DiffusionNetAutoencoder


def area_weighted_mse(pred, target, mass):
    """
    Area-weighted MSE loss. This makes the loss discretization-invariant:
    denser regions don't dominate just because they have more vertices.

    Args:
        pred:   (V, C) predicted features
        target: (V, C) target features
        mass:   (V,) area weights
    Returns:
        scalar loss
    """
    diff_sq = (pred - target) ** 2  # (V, C)
    weighted = diff_sq * mass.unsqueeze(-1)  # (V, C)
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
        mass = sample['mass'].to(device)
        L = sample['L'].to(device)
        evals = sample['evals'].to(device)
        evecs = sample['evecs'].to(device)
        gradX = sample['gradX'].to(device)
        gradY = sample['gradY'].to(device)

        optimizer.zero_grad()
        hks_recon, z = model(hks, mass, L, evals, evecs, gradX, gradY)
        loss = area_weighted_mse(hks_recon, hks, mass)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_samples += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, dataset, indices, device):
    """Evaluate on the given indices. Returns avg loss and all latent codes."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    latents = []
    ids = []

    for idx in indices:
        sample = dataset[idx]

        hks = sample['hks'].to(device)
        mass = sample['mass'].to(device)
        L = sample['L'].to(device)
        evals = sample['evals'].to(device)
        evecs = sample['evecs'].to(device)
        gradX = sample['gradX'].to(device)
        gradY = sample['gradY'].to(device)

        hks_recon, z = model(hks, mass, L, evals, evecs, gradX, gradY)
        loss = area_weighted_mse(hks_recon, hks, mass)

        total_loss += loss.item()
        n_samples += 1
        latents.append(z.cpu().numpy())
        ids.append(sample['meta']['id'])

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, np.array(latents), ids


@torch.no_grad()
def extract_all_latents(model, dataset, device):
    """Extract latent codes for the entire dataset."""
    model.eval()
    latents = []
    ids = []
    timepoints = []
    areas = []

    for idx in tqdm(range(len(dataset)), desc="Extracting latents"):
        sample = dataset[idx]

        hks = sample['hks'].to(device)
        mass = sample['mass'].to(device)
        L = sample['L'].to(device)
        evals = sample['evals'].to(device)
        evecs = sample['evecs'].to(device)
        gradX = sample['gradX'].to(device)
        gradY = sample['gradY'].to(device)

        _, z = model(hks, mass, L, evals, evecs, gradX, gradY)

        latents.append(z.cpu().numpy())
        ids.append(sample['meta']['id'])
        timepoints.append(sample['meta']['timepoint'])
        areas.append(sample['area'])

    return np.array(latents), ids, timepoints, np.array(areas)


def main():
    parser = argparse.ArgumentParser(description="Train DiffusionNet autoencoder")
    parser.add_argument('--data_path', type=str, default='Data/20260224',
                        help='Path to dataset root')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (default: {data_path}/config.json)')
    parser.add_argument('--output_dir', type=str, default='ML/outputs',
                        help='Directory for saved models and embeddings')
    parser.add_argument('--op_cache_dir', type=str, default='ML/op_cache',
                        help='Directory to cache DiffusionNet operators')

    # Model hyperparameters
    parser.add_argument('--k_eig', type=int, default=128,
                        help='Number of eigenvalues for DiffusionNet')
    parser.add_argument('--hks_scales_path', type=str, default='sim/vocab_new.npz',
                        help='Path to .npz with HKS time scales (ts array)')
    parser.add_argument('--C_latent', type=int, default=32,
                        help='Latent space dimension')
    parser.add_argument('--C_width', type=int, default=64,
                        help='Internal DiffusionNet width')
    parser.add_argument('--N_block_enc', type=int, default=3,
                        help='Number of encoder blocks')
    parser.add_argument('--N_block_dec', type=int, default=2,
                        help='Number of decoder blocks')
    parser.add_argument('--no_gradient_features', action='store_true',
                        help='Disable gradient features')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (0 = train on all)')
    parser.add_argument('--quality_percentile', type=float, default=95,
                        help='Percentile threshold for filtering bad meshes')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Limit dataset to first N samples (0 = use all)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(args.data_path, 'config.json')

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset ---
    print("Loading dataset...")
    dataset = OrganoidDataset(
        data_path=args.data_path,
        config_path=args.config,
        k_eig=args.k_eig,
        hks_scales_path=args.hks_scales_path,
        op_cache_dir=args.op_cache_dir,
        recon_quality_threshold=args.quality_percentile,
        preload=True,
    )

    if args.max_samples > 0:
        dataset.entries = dataset.entries[:args.max_samples]
        print(f"Truncated to {len(dataset.entries)} samples")

    n_hks = dataset.n_hks  # read from the scales file

    # --- Model ---
    model = DiffusionNetAutoencoder(
        C_in=n_hks,
        C_latent=args.C_latent,
        C_width=args.C_width,
        N_block_enc=args.N_block_enc,
        N_block_dec=args.N_block_dec,
        mlp_hidden_dims=[args.C_width],
        dropout=False,
        with_gradient_features=not args.no_gradient_features,
        with_gradient_rotations=not args.no_gradient_features,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # --- Training ---
    all_indices = np.arange(len(dataset))

    if args.n_folds > 1:
        # K-fold cross-validation
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
            print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")

            # Re-init model for each fold
            model = DiffusionNetAutoencoder(
                C_in=n_hks,
                C_latent=args.C_latent,
                C_width=args.C_width,
                N_block_enc=args.N_block_enc,
                N_block_dec=args.N_block_dec,
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
                    val_loss, _, _ = evaluate(model, dataset, val_idx, device)
                    print(f"  Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(),
                                   os.path.join(args.output_dir, f'model_fold{fold}.pt'))

            fold_results.append(best_val_loss)
            print(f"  Best val loss: {best_val_loss:.4f}")

        print(f"\nCV results: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")

    # --- Final model: train on all data ---
    print("\n=== Training final model on all data ===")
    model = DiffusionNetAutoencoder(
        C_in=n_hks,
        C_latent=args.C_latent,
        C_width=args.C_width,
        N_block_enc=args.N_block_enc,
        N_block_dec=args.N_block_dec,
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
            print(f"  Epoch {epoch:3d} | loss: {train_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'model_final.pt'))

    # --- Extract and save latent embeddings ---
    print("\nExtracting latent embeddings...")
    latents, ids, timepoints, areas = extract_all_latents(model, dataset, device)

    np.savez(
        os.path.join(args.output_dir, 'latent_embeddings.npz'),
        latents=latents,
        ids=np.array(ids),
        timepoints=np.array(timepoints),
        areas=areas,
    )
    print(f"Saved latent embeddings: shape {latents.shape}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
