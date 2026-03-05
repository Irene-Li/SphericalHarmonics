"""
Training script for the OrganoidVAE with SphereDecoder.

Loss = Chamfer_Procrustes(pred_mesh_verts, target_verts)
     + beta_kl * KL( q(z|x) || N(0,I) )

Chamfer_Procrustes: optimally rotates the predicted point cloud to align
    with the target (closed-form SVD) before measuring Chamfer distance.
    This is differentiable and removes the need for a learned pose encoder.

The decoder outputs a deformed icosphere mesh (B, V_sphere, 3).  The sphere
template vertices and faces are cached once and passed at each forward call.

Usage:
    python DiffusionML/experiments/hks_autoencoder/train.py \
        --data_path Data/small_meshes --epochs 300 \
        --beta_kl 1e-3 --beta_warmup 100 \
        --lr_scheduler cosine --lr_min 1e-5

Latent embeddings (posterior means mu) are saved after training for
downstream clustering / visualisation.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dataset import OrganoidDataset, collate_single
from model import OrganoidVAE, kl_divergence, procrustes_rotation


# ---------------------------------------------------------------------------
# Geometry utilities (not in diffusion_net core)
# ---------------------------------------------------------------------------

def sample_sphere_points(n_points: int, device: torch.device) -> torch.Tensor:
    """
    Sample n_points uniformly on the unit sphere surface.

    Uses the standard trick: draw from an isotropic Gaussian and normalise
    each point to the sphere.  This gives a uniform distribution on S².

    Returns (n_points, 3) float32 tensor on the given device.
    """
    pts = torch.randn(n_points, 3, device=device)
    pts = pts / pts.norm(dim=1, keepdim=True)
    return pts


def chamfer_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer distance between two point clouds.

    A: (N, 3), B: (M, 3)
    Returns scalar = mean(A→B nearest) + mean(B→A nearest).
    """
    d = torch.cdist(A, B)  # (N, M)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def generate_icosphere(subdivisions: int = 4):
    """
    Generate a unit icosphere by recursive midpoint subdivision.

    Returns (verts, faces) as float32 / int64 tensors.
    subdivisions=4 → 2562 vertices, 5120 faces.
    """
    import numpy as _np
    phi = (1.0 + _np.sqrt(5.0)) / 2.0
    v = _np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=_np.float64)
    v = v / _np.linalg.norm(v, axis=1, keepdims=True)
    f = _np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=_np.int64)

    for _ in range(subdivisions):
        new_f, mid = [], {}
        def _mid(a, b):
            nonlocal v
            key = (min(a,b), max(a,b))
            if key not in mid:
                m = (v[a] + v[b]) / 2.0
                mid[key] = len(v)
                v = _np.vstack([v, m / _np.linalg.norm(m)])
            return mid[key]
        for a, b, c in f:
            ab, bc, ca = _mid(a,b), _mid(b,c), _mid(c,a)
            new_f += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        f = _np.array(new_f, dtype=_np.int64)

    return (torch.tensor(v, dtype=torch.float32),
            torch.tensor(f, dtype=torch.int64))


def load_sphere_template(subdivisions, device, output_dir):
    """
    Load or create the icosphere template, caching it to disk.

    Returns sphere_verts (V, 3) and sphere_faces (F, 3) on the given device.
    """
    cache_path = os.path.join(output_dir, 'sphere_template.pt')
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location=device)
        return data['verts'].to(device), data['faces'].to(device)

    verts, faces = generate_icosphere(subdivisions=subdivisions)
    torch.save({'verts': verts, 'faces': faces}, cache_path)
    return verts.to(device), faces.to(device)


def chamfer_procrustes(pred_verts, target_verts):
    """
    Procrustes-aligned Chamfer distance (translation + O(3) rotation/reflection).

    Both point clouds are centred at the origin, then the optimal orthogonal
    transformation R ∈ O(3) (rotations AND reflections) is applied to pred
    before measuring Chamfer distance.  Allowing reflections lets the solver
    find the globally best alignment even when pred and target are mirror
    images of each other.

    The SVD alignment uses nearest-neighbour correspondences (standard
    Procrustes on unordered point clouds).  Near-spherical predictions can
    produce noisy R when singular values are near-equal, but balanced-bin
    epoch sampling (BalancedBinSampler) substantially reduces the frequency
    of near-sphere samples seen per epoch, making the gradient signal stable
    enough in practice.

    pred_verts:   (N_pred,   3)
    target_verts: (N_target, 3)
    Returns scalar.
    """
    pred_c   = pred_verts   - pred_verts.mean(dim=0)
    target_c = target_verts - target_verts.mean(dim=0)
    # Alignment is a pre-processing step — we do NOT want gradients to flow
    # through the SVD, only through the subsequent Chamfer distance.
    with torch.no_grad():
        R = procrustes_rotation(pred_c, target_c, allow_flips=True)
    pred_aligned = pred_c @ R.T
    return chamfer_distance(pred_aligned, target_c)


def compute_loss(pred_verts, target_verts, mu, log_var, beta_kl=1e-3):
    """
    Full training loss for the SphereDecoder VAE.

    pred_verts:   (V_sphere, 3)  deformed sphere mesh (squeezed from batch)
    target_verts: (V_orig,   3)  original organoid vertices
    mu, log_var:  (B, C_latent)

    Returns: total, chamfer, kl
    """
    chamfer = chamfer_procrustes(pred_verts, target_verts)
    kl      = kl_divergence(mu, log_var)
    total   = chamfer + beta_kl * kl
    return total, chamfer, kl


def get_beta(epoch: int, beta_kl: float, warmup_epochs: int) -> float:
    """
    Linear beta-KL warmup: ramp from 0 → beta_kl over the first
    `warmup_epochs` epochs, then hold at beta_kl.

    This lets the reconstruction loss converge before the KL penalty
    kicks in, preventing premature posterior collapse.
    """
    if warmup_epochs <= 0:
        return beta_kl
    return beta_kl * min(1.0, epoch / warmup_epochs)


def make_scheduler(optimizer, scheduler_type: str, n_epochs: int, lr_min: float):
    """
    Build an LR scheduler.
      'cosine'  — CosineAnnealingLR: smooth decay from lr → lr_min over n_epochs
      'plateau' — ReduceLROnPlateau: halve lr when val loss stalls (patience=20)
      'none'    — no scheduler
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr_min)
    if scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    return None


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

def train_one_epoch(model, sphere_verts, dataset, indices, optimizer, device,
                    epoch, beta_kl, scheduler=None):
    """Train for one epoch, one optimizer step per sample.

    Each mesh has a variable number of vertices, so true mini-batching is not
    feasible.  We perform a forward/backward/step for every sample individually.
    """
    model.train()
    totals = dict(loss=0., chamfer=0., kl=0.)
    n_samples = 0

    pbar = tqdm(enumerate(indices), total=len(indices),
                desc=f"  Epoch {epoch}", leave=False)
    for step, idx in pbar:
        sample = dataset[idx]
        hks   = _to_device(sample['hks'],   device)
        verts = _to_device(sample['verts'], device)
        mass  = _to_device(sample['mass'],  device)
        L     = _to_device(sample['L'],     device)
        evals = _to_device(sample['evals'], device)
        evecs = _to_device(sample['evecs'], device)
        gradX = _to_device(sample['gradX'], device)
        gradY = _to_device(sample['gradY'], device)

        # Sample fresh random points on S² for this step (AtlasNet training
        # convention).  The decoder sees a different random set of sphere
        # points each forward pass, forcing it to learn a smooth continuous
        # function over S² rather than memorising per-vertex offsets.
        # sphere_verts (fixed icosphere) is used only during evaluation.
        train_sphere = sample_sphere_points(sphere_verts.shape[0], device)

        optimizer.zero_grad()
        points, mu, log_var = model(
            hks, train_sphere, mass, L, evals, evecs, gradX, gradY)
        pts_sq = points.squeeze(0)   # (V_sphere, 3)

        loss, chamfer, kl = compute_loss(
            pts_sq, verts, mu, log_var, beta_kl=beta_kl)

        loss.backward()
        optimizer.step()

        totals['loss']    += loss.item()
        totals['chamfer'] += chamfer.item()
        totals['kl']      += kl.item()
        n_samples += 1

        pbar.set_postfix(
            loss=f"{loss.item():.5f}",
            cd=f"{chamfer.item():.5f}",
            kl=f"{kl.item():.4f}",
            beta=f"{beta_kl:.2e}",
        )

    # Cosine scheduler steps once per epoch
    if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()

    n = max(n_samples, 1)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, sphere_verts, dataset, indices, device, beta_kl):
    model.eval()
    totals = dict(loss=0., chamfer=0., kl=0.)
    n_samples = 0
    latents, ids = [], []

    for idx in indices:
        sample = dataset[idx]
        hks   = _to_device(sample['hks'],   device)
        verts = _to_device(sample['verts'], device)
        mass  = _to_device(sample['mass'],  device)
        L     = _to_device(sample['L'],     device)
        evals = _to_device(sample['evals'], device)
        evecs = _to_device(sample['evecs'], device)
        gradX = _to_device(sample['gradX'], device)
        gradY = _to_device(sample['gradY'], device)

        points, mu, log_var = model(
            hks, sphere_verts, mass, L, evals, evecs, gradX, gradY)
        pts_sq = points.squeeze(0)

        loss, chamfer, kl = compute_loss(
            pts_sq, verts, mu, log_var, beta_kl=beta_kl)
        totals['loss']    += loss.item()
        totals['chamfer'] += chamfer.item()
        totals['kl']      += kl.item()
        n_samples += 1
        latents.append(mu.cpu().numpy())
        ids.append(sample['meta']['id'])

    n = max(n_samples, 1)
    return {k: v / n for k, v in totals.items()}, np.array(latents), ids


@torch.no_grad()
def extract_all_latents(model, sphere_verts, dataset, device):
    model.eval()
    latents, ids, timepoints = [], [], []
    for idx in tqdm(range(len(dataset)), desc="Extracting latents"):
        sample = dataset[idx]
        hks   = _to_device(sample['hks'],   device)
        mass  = _to_device(sample['mass'],  device)
        L     = _to_device(sample['L'],     device)
        evals = _to_device(sample['evals'], device)
        evecs = _to_device(sample['evecs'], device)
        gradX = _to_device(sample['gradX'], device)
        gradY = _to_device(sample['gradY'], device)

        _, mu, _ = model(hks, sphere_verts, mass, L, evals, evecs, gradX, gradY)
        latents.append(mu.cpu().numpy())
        ids.append(sample['meta']['id'])
        timepoints.append(sample['meta']['timepoint'])

    return np.array(latents), ids, timepoints


def _make_model(args, n_hks, device):
    return OrganoidVAE(
        C_in=n_hks,
        C_latent=args.C_latent,
        C_fate=args.C_fate,
        C_width=args.C_width,
        dec_width=args.dec_width,
        dec_layers=args.dec_layers,
    ).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train OrganoidVAE (SphereDecoder)")
    parser.add_argument('--data_path',    type=str,   default='Data/small_meshes')
    parser.add_argument('--output_dir',   type=str,
                        default='DiffusionML/experiments/hks_autoencoder/outputs')
    parser.add_argument('--op_cache_dir', type=str,   default='DiffusionML/op_cache')

    # Encoder
    parser.add_argument('--k_eig',    type=int,   default=128)
    parser.add_argument('--C_latent', type=int,   default=32)
    parser.add_argument('--C_width',  type=int,   default=64)
    parser.add_argument('--C_fate',   type=int,   default=0)

    # Decoder (SphereDecoder)
    parser.add_argument('--dec_width',       type=int, default=128)
    parser.add_argument('--dec_layers',      type=int, default=4)
    parser.add_argument('--sphere_subdiv',   type=int, default=4,
                        help='Icosphere subdivision level (4 → 2562 vertices)')

    # Loss
    parser.add_argument('--beta_kl', type=float, default=1e-3,
                        help='Target KL weight (beta-VAE).')
    parser.add_argument('--beta_warmup', type=int, default=100,
                        help='Epochs to linearly ramp beta from 0 → beta_kl (0 = no warmup).')

    # Training
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--lr_min',       type=float, default=1e-5,
                        help='Minimum LR for cosine scheduler.')
    parser.add_argument('--lr_scheduler', type=str,   default='cosine',
                        choices=['none', 'cosine', 'plateau'],
                        help='LR scheduler type.')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_folds',      type=int,   default=5)
    parser.add_argument('--max_samples',  type=int,   default=0)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--checkpoint_every', type=int, default=25,
                        help='Save model_checkpoint.pt every N epochs during final '
                             'training (0 = disabled). Allows inspection mid-run.')
    parser.add_argument('--sphere_cd_threshold', type=float, default=0.05,
                        help='Sphere-CD threshold for balanced epoch sampling. '
                             'Meshes below this value are "spherical", those at or '
                             'above are "irregular"; each epoch draws equally from '
                             'both buckets. Set to 0 to disable (default: 0.05).')
    parser.add_argument('--corrected_only', action='store_true',
                        help='Only train on meshes whose filename starts with "N" '
                             '(hand-corrected meshes). All other files are ignored.')

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

    # Load sphere template (cached)
    sphere_verts, sphere_faces = load_sphere_template(
        args.sphere_subdiv, device, args.output_dir)
    print(f"Sphere template: {sphere_verts.shape[0]} vertices, "
          f"{sphere_faces.shape[0]} faces (subdiv={args.sphere_subdiv})")

    print("Loading dataset...")
    dataset = OrganoidDataset(
        data_path=args.data_path,
        k_eig=args.k_eig,
        op_cache_dir=args.op_cache_dir,
        sphere_cd_threshold=args.sphere_cd_threshold,
        corrected_only=args.corrected_only,
    )
    if args.max_samples > 0:
        dataset.entries = dataset.entries[:args.max_samples]
        print(f"Truncated to {len(dataset.entries)} samples")

    n_hks = dataset.n_hks
    model = _make_model(args, n_hks, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"  Encoder : DiffusionNet C_width={args.C_width} -> (mu, log_var) C_latent={args.C_latent}")
    print(f"  Decoder : SphereDecoder width={args.dec_width} layers={args.dec_layers}, "
          f"output {sphere_verts.shape[0]} verts")
    print(f"  Loss    : Procrustes-Chamfer + β*KL  (β warms up over {args.beta_warmup} epochs → {args.beta_kl})")
    print(f"  LR scheduler: {args.lr_scheduler}  lr={args.lr} → {args.lr_min}")

    all_indices = np.arange(len(dataset))

    if args.n_folds > 1:
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
            print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
            model = _make_model(args, n_hks, device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
            scheduler = make_scheduler(optimizer, args.lr_scheduler,
                                       args.epochs, args.lr_min)
            best_val_loss = float('inf')

            for epoch in range(1, args.epochs + 1):
                beta = get_beta(epoch, args.beta_kl, args.beta_warmup)
                if dataset.sphere_sampler is not None:
                    epoch_train = dataset.sphere_sampler.sample_epoch(train_idx)
                else:
                    np.random.shuffle(train_idx)
                    epoch_train = train_idx
                tr = train_one_epoch(model, sphere_verts, dataset, epoch_train,
                                     optimizer, device, epoch, beta, scheduler)
                if epoch % 10 == 0 or epoch == args.epochs:
                    val, _, _ = evaluate(model, sphere_verts, dataset, val_idx,
                                         device, beta)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch:3d} | β={beta:.2e} lr={current_lr:.2e} | "
                          f"train {tr['loss']:.5f} (CD={tr['chamfer']:.5f} KL={tr['kl']:.4f}) | "
                          f"val {val['loss']:.5f} (CD={val['chamfer']:.5f} KL={val['kl']:.4f})")
                    if val['loss'] < best_val_loss:
                        best_val_loss = val['loss']
                        torch.save(model.state_dict(),
                                   os.path.join(args.output_dir, f'model_fold{fold}.pt'))
                    if args.lr_scheduler == 'plateau':
                        scheduler.step(val['loss'])

            fold_results.append(best_val_loss)
            print(f"  Best val loss: {best_val_loss:.5f}")

        print(f"\nCV: {np.mean(fold_results):.5f} ± {np.std(fold_results):.5f}")

    print("\n=== Training final model on all data ===")
    model = _make_model(args, n_hks, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.lr_scheduler,
                               args.epochs, args.lr_min)

    for epoch in range(1, args.epochs + 1):
        beta = get_beta(epoch, args.beta_kl, args.beta_warmup)
        if dataset.sphere_sampler is not None:
            epoch_indices = dataset.sphere_sampler.sample_epoch(all_indices)
        else:
            np.random.shuffle(all_indices)
            epoch_indices = all_indices
        tr = train_one_epoch(model, sphere_verts, dataset, epoch_indices,
                             optimizer, device, epoch, beta, scheduler)
        if epoch % 10 == 0 or epoch == args.epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | β={beta:.2e} lr={current_lr:.2e} | "
                  f"loss={tr['loss']:.5f}  CD={tr['chamfer']:.5f}  KL={tr['kl']:.4f}")

        # Periodic checkpoint so the notebook can inspect the model mid-run
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.output_dir, 'model_checkpoint.pt')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'chamfer': tr['chamfer'], 'kl': tr['kl']}, ckpt_path)

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'model_final.pt'))

    config = vars(args)
    config['n_hks'] = int(n_hks)
    config['sphere_n_verts'] = int(sphere_verts.shape[0])
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("\nExtracting latent embeddings...")
    latents, ids, timepoints = extract_all_latents(model, sphere_verts, dataset, device)
    np.savez(
        os.path.join(args.output_dir, 'latent_embeddings.npz'),
        latents=latents,
        ids=np.array(ids),
        timepoints=np.array(timepoints),
    )
    print(f"Saved embeddings: shape {latents.shape}")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
