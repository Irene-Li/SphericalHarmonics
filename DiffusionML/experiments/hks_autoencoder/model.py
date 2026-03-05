"""
DiffusionNet variational autoencoder for organoid shape embedding.

Architecture:
    Encoder : DiffusionNet -> global mean pool -> fc_mu + fc_log_var
    Decoder : AtlasNet-style MLP (Groueix et al. 2018), single-patch case.
              sphere_coords → Conv1d(3, C_width) + proj_z(z) → ReLU
              → [Conv1d(C_width, C_width) → ReLU] × (n_layers-1)
              → Conv1d(C_width, 3)
              z enters once as an additive bias after the first conv — this
              is mathematically equivalent to concatenating z with every
              input point but avoids the concatenation overhead.
              During training sphere_verts are randomly sampled on S²
              (regularisation); the fixed icosphere is used at eval time.
              Output is (B, V_sphere, 3).

Pose handling:
    Translation removed by centring both clouds before Chamfer.
    HKS is intrinsically rotation-invariant so the encoder produces
    orientation-independent codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_net.layers import DiffusionNet


def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    (Zhou et al. 2019, 'On the Continuity of Rotation Representations in Deep Learning')
    """
    a1, a2 = d6[:, 0:3], d6[:, 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)  # (B, 3, 3)


def procrustes_rotation(A, B, allow_flips=True):
    """
    Compute the optimal orthogonal transformation R* that aligns point cloud A
    to B, minimising  ||A_c R* - B_nn||_F  where B_nn contains the
    nearest-neighbour match in B for each point in A.

    A: (N_A, 3) — predicted point cloud (may differ in size from B)
    B: (N_B, 3) — target point cloud / mesh vertices
    allow_flips: if True (default), R* ∈ O(3) — reflections are allowed and
                 det(R*) may be -1.  This lets the solver find the globally
                 best alignment even when the two shapes are mirror images.
                 If False, R* is constrained to SO(3) (det = +1) via the
                 standard sign-correction on the smallest singular value.
    Returns R* as a (3, 3) tensor.

    Derivation: SVD of cross-covariance H = A_c^T B_nn
      H = U S Vh  →  R* = Vh^T U^T  (O(3))
      with optional sign correction for SO(3).
    """
    # Centre both clouds independently
    A_c = A - A.mean(dim=0, keepdim=True)   # (N_A, 3)
    B_c = B - B.mean(dim=0, keepdim=True)   # (N_B, 3)

    # Nearest-neighbour correspondences: for each point in A_c find closest in B_c
    nn_idx = torch.cdist(A_c, B_c).argmin(dim=1)   # (N_A,)
    B_nn   = B_c[nn_idx]                            # (N_A, 3)

    # Cross-covariance matrix  (3, 3)
    H = A_c.T @ B_nn

    U, S, Vh = torch.linalg.svd(H)

    if allow_flips:
        # O(3): allow det = ±1 — reflections accepted
        R = Vh.T @ U.T
    else:
        # SO(3): flip sign of the last column of Vh when det would be -1
        d = torch.linalg.det(Vh.T @ U.T)
        D = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=A.device))
        R = Vh.T @ D @ U.T

    return R   # (3, 3)


# ---------------------------------------------------------------------------
#  AtlasNet-style decoder on sphere template
# ---------------------------------------------------------------------------

class SphereDecoder(nn.Module):
    """
    AtlasNet-style MLP decoder (Groueix et al. 2018), single-patch case.

    Follows the original architecture exactly:
      - Conv1d(3, C_width) applied to sphere coords (channels-first layout)
      - Latent z projected to C_width and added as an additive bias,
        broadcast across all V vertices.  This is equivalent to concatenating
        z with every input point (as noted in the AtlasNet codebase) but
        cheaper: only one linear projection of z is needed rather than
        repeating z for every vertex before the first layer.
      - ReLU after each layer except the last.
      - Conv1d(C_width, 3) final projection, no activation.

    During training, sphere_verts should be random points sampled uniformly
    on S² (see sample_sphere_points in train.py).  This forces the decoder
    to learn a smooth continuous function over the sphere rather than
    memorising per-vertex offsets, acting as free regularisation.
    The fixed icosphere is used at evaluation time.
    """

    def __init__(self, C_latent, C_fate=0, C_width=128, n_layers=4):
        super().__init__()
        z_dim = C_latent + C_fate

        # Project latent to decoder width for the additive injection.
        # No bias: the bias in conv1 absorbs the constant offset.
        self.proj_z = nn.Linear(z_dim, C_width, bias=False)

        # First layer: sphere coords (3) → C_width features
        self.conv1 = nn.Conv1d(3, C_width, 1)

        # Hidden layers: C_width → C_width  (n_layers - 1 of them)
        self.convs = nn.ModuleList(
            [nn.Conv1d(C_width, C_width, 1) for _ in range(n_layers - 1)])

        # Final projection: C_width → 3, no activation
        self.final_head = nn.Conv1d(C_width, 3, 1)

    def forward(self, z, sphere_verts, cell_fate=None):
        """
        Args:
            z:            (B, C_latent)
            sphere_verts: (V, 3)  sphere points — random during training,
                          fixed icosphere at eval time
            cell_fate:    (B, C_fate) optional
        Returns:
            (B, V, 3)  output point cloud
        """
        B = z.shape[0]
        z_combined = z if cell_fate is None else torch.cat([z, cell_fate], dim=-1)

        # Project latent → (B, C_width, 1), broadcast over V by Conv1d semantics
        z_proj = self.proj_z(z_combined).unsqueeze(-1)          # (B, C_width, 1)

        # Sphere coords: (V, 3) → (B, 3, V)  (channels-first for Conv1d)
        sv = sphere_verts.t().unsqueeze(0).expand(B, -1, -1)    # (B, 3, V)

        # First layer: conv(sphere_coords) + z broadcast as additive bias
        h = F.relu(self.conv1(sv) + z_proj)                     # (B, C_width, V)

        # Hidden layers
        for conv in self.convs:
            h = F.relu(conv(h))                                  # (B, C_width, V)

        # Final projection: (B, C_width, V) → (B, V, 3)
        return self.final_head(h).permute(0, 2, 1)              # (B, V, 3)


# ---------------------------------------------------------------------------

def kl_divergence(mu, log_var):
    """
    Closed-form KL( N(mu, sigma^2) || N(0, 1) ) summed over the latent
    dimension and averaged over the batch.

    KL = -0.5 * sum_d [ 1 + log_var_d - mu_d^2 - exp(log_var_d) ]
    """
    return -0.5 * torch.mean(
        torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    )


class OrganoidVAE(nn.Module):
    """
    Variational autoencoder for organoid shape embedding.

    Encoder: DiffusionNet (HKS) -> global mean pool -> (mu, log_var)
    Decoder: AtlasNet-style MLP (single patch).
             conv(sphere_coords) + proj(z) -> ReLU hidden layers -> 3-D
             Output is (B, V_sphere, 3).

    At eval time reparameterize() returns mu directly as the deterministic,
    rotation-invariant shape embedding.
    """

    def __init__(self, C_in=16, C_latent=64, C_fate=0, C_width=128,
                 dec_width=128, dec_layers=4):
        super().__init__()
        self.C_latent = C_latent

        # Encoder backbone
        self.shape_encoder = DiffusionNet(
            C_in=C_in, C_out=C_width, C_width=C_width,
            outputs_at='global_mean',
        )
        self.fc_mu      = nn.Linear(C_width, C_latent)
        self.fc_log_var = nn.Linear(C_width, C_latent)

        # Decoder (icosphere template passed at forward time)
        self.decoder = SphereDecoder(
            C_latent=C_latent, C_fate=C_fate,
            C_width=dec_width, n_layers=dec_layers,
        )

    def encode(self, x_hks, mass, L, evals, evecs, gradX, gradY):
        """Returns (mu, log_var), each (B, C_latent)."""
        h = self.shape_encoder(x_hks, mass, L, evals, evecs, gradX, gradY)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu

    def forward(self, x_hks, sphere_verts, mass, L, evals, evecs, gradX, gradY,
                fate_vec=None):
        """
        Args:
            x_hks:       (V_mesh, C_in)  HKS features on the input mesh
            sphere_verts: (V_sphere, 3)  icosphere template vertices
            mass, L, evals, evecs, gradX, gradY: DiffusionNet operators
            fate_vec:    (B, C_fate) optional cell-fate conditioning

        Returns:
            points:  (B, V_sphere, 3)  deformed sphere mesh (canonical pose)
            mu:      (B, C_latent)     posterior mean — use as embedding
            log_var: (B, C_latent)     posterior log-variance
        """
        mu, log_var = self.encode(x_hks, mass, L, evals, evecs, gradX, gradY)
        z      = self.reparameterize(mu, log_var)
        points = self.decoder(z, sphere_verts, cell_fate=fate_vec)
        return points, mu, log_var
