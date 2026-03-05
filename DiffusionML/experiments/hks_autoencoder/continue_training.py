"""
Continue training an OrganoidVAE from a saved model_final.pt checkpoint.

Reads config.json from the same output directory to reconstruct the model
architecture automatically — no need to re-specify architecture flags.

Usage:
    python continue_training.py --output_dir outputs/ --epochs 200

    # Override LR (recommended — default is 1e-4, lower than the original 1e-3)
    python continue_training.py --output_dir outputs/ --epochs 200 --lr 5e-4

    # Resume from a specific weights file in a different location
    python continue_training.py --output_dir outputs/ --epochs 200 \
        --weights path/to/model_final.pt
"""

import os
import sys
import argparse
import json
import types
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dataset import OrganoidDataset
from model import OrganoidVAE
from train import (
    train_one_epoch, evaluate, extract_all_latents,
    load_sphere_template, make_scheduler, get_beta,
)


def save_checkpoint(path, epoch, model, optimizer, scheduler, args, metrics):
    """Save a full training checkpoint (model + optimizer + scheduler state)."""
    torch.save({
        'epoch':      epoch,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict() if scheduler is not None else None,
        'args':       vars(args) if not isinstance(args, dict) else args,
        'chamfer':    metrics.get('chamfer'),
        'kl':         metrics.get('kl'),
    }, path)


def load_config(output_dir):
    config_path = os.path.join(output_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found in {output_dir}. "
            "This file is saved automatically at the end of training.")
    with open(config_path) as f:
        return json.load(f)


def config_to_args(config, overrides):
    """
    Merge config.json values with command-line overrides into a simple
    namespace, so the rest of the code can treat it like argparse output.
    """
    merged = dict(config)
    merged.update({k: v for k, v in vars(overrides).items()
                   if v is not None or k not in merged})
    return types.SimpleNamespace(**merged)


def main():
    parser = argparse.ArgumentParser(
        description="Continue training OrganoidVAE from model_final.pt")
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing model_final.pt and config.json.')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to weights file. Defaults to '
                             '<output_dir>/model_final.pt.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of *additional* epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate. Defaults to 1e-4 (lower than the '
                             'original 1e-3, appropriate for fine-tuning).')
    parser.add_argument('--lr_min', type=float, default=None,
                        help='Minimum LR for cosine scheduler. '
                             'Defaults to lr / 10.')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=['none', 'cosine', 'plateau'],
                        help='LR scheduler. Defaults to the value in config.json.')
    parser.add_argument('--beta_kl', type=float, default=None,
                        help='KL weight. Defaults to the value in config.json.')
    parser.add_argument('--beta_warmup', type=int, default=0,
                        help='KL warmup epochs. Default 0 (no warmup — model is '
                             'already trained, no need to ramp again).')
    parser.add_argument('--checkpoint_every', type=int, default=None,
                        help='Save checkpoint every N epochs. '
                             'Defaults to the value in config.json.')
    parser.add_argument('--seed', type=int, default=42)

    cli = parser.parse_args()

    # ── Load config and merge with CLI overrides ──────────────────────────
    config = load_config(cli.output_dir)
    args = config_to_args(config, cli)

    # Fill in defaults that depend on other values
    if cli.lr_min is None:
        args.lr_min = cli.lr / 10.0
    args.lr = cli.lr
    args.beta_warmup = cli.beta_warmup
    args.epochs_additional = cli.epochs   # store for display

    weights_path = cli.weights or os.path.join(cli.output_dir, 'model_final.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── Sphere template ───────────────────────────────────────────────────
    sphere_verts, sphere_faces = load_sphere_template(
        args.sphere_subdiv, device, cli.output_dir)
    print(f"Sphere template: {sphere_verts.shape[0]} verts "
          f"(subdiv={args.sphere_subdiv})")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = OrganoidDataset(
        data_path=args.data_path,
        k_eig=args.k_eig,
        op_cache_dir=args.op_cache_dir,
    )
    all_indices = np.arange(len(dataset))
    print(f"Dataset: {len(dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────────
    model = OrganoidVAE(
        C_in=args.n_hks,
        C_latent=args.C_latent,
        C_fate=args.C_fate,
        C_width=args.C_width,
        dec_width=args.dec_width,
        dec_layers=args.dec_layers,
    ).to(device)

    print(f"Loading weights from: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)
    # Handle both plain state dict and full checkpoint formats
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print("Weights loaded successfully.")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.lr_scheduler,
                               cli.epochs, args.lr_min)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Continuing for {cli.epochs} additional epochs")
    print(f"LR: {args.lr} → {args.lr_min}  scheduler: {args.lr_scheduler}")
    print(f"β_KL: {args.beta_kl}  warmup: {args.beta_warmup} epochs")

    # ── Training loop ─────────────────────────────────────────────────────
    # Epoch numbers start at 1 here (they represent epochs within this
    # continuation run, not absolute epoch count).
    tr = {}
    for epoch in range(1, cli.epochs + 1):
        beta = get_beta(epoch, args.beta_kl, args.beta_warmup)
        np.random.shuffle(all_indices)
        tr = train_one_epoch(model, sphere_verts, dataset, all_indices,
                             optimizer, device, epoch, beta, scheduler)

        if epoch % 10 == 0 or epoch == cli.epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d}/{cli.epochs} | "
                  f"β={beta:.2e} lr={current_lr:.2e} | "
                  f"loss={tr['loss']:.5f}  "
                  f"CD={tr['chamfer']:.5f}  "
                  f"KL={tr['kl']:.4f}")

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(cli.output_dir, 'model_checkpoint.pt')
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler,
                            args, tr)
            print(f"  Checkpoint saved: {ckpt_path}")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(model.state_dict(),
               os.path.join(cli.output_dir, 'model_final.pt'))
    save_checkpoint(
        os.path.join(cli.output_dir, 'model_final_checkpoint.pt'),
        cli.epochs, model, optimizer, scheduler, args, tr)
    print(f"\nSaved: {cli.output_dir}/model_final.pt")

    # ── Extract latents ───────────────────────────────────────────────────
    print("\nExtracting latent embeddings...")
    latents, ids, timepoints = extract_all_latents(
        model, sphere_verts, dataset, device)
    np.savez(
        os.path.join(cli.output_dir, 'latent_embeddings.npz'),
        latents=latents,
        ids=np.array(ids),
        timepoints=np.array(timepoints),
    )
    print(f"Saved embeddings: shape {latents.shape}")


if __name__ == '__main__':
    main()
