"""
DiffusionNet autoencoder for organoid shape embedding.

Architecture:
    Encoder: DiffusionNet blocks -> area-weighted global mean pooling -> latent vector
    Decoder: MLP broadcast to vertices -> DiffusionNet blocks -> per-vertex output

The encoder produces a rotationally-invariant shape descriptor (the latent code).
The decoder reconstructs per-vertex HKS from the latent code.
"""

import torch
import torch.nn as nn

from diffusion_net.layers import DiffusionNet


class DiffusionNetEncoder(nn.Module):
    """
    Encoder: per-vertex features -> global latent vector.
    Uses DiffusionNet with global_mean output.
    """

    def __init__(self, C_in, C_latent, C_width=64, N_block=3,
                 mlp_hidden_dims=None, dropout=False,
                 with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width]

        self.diffnet = DiffusionNet(
            C_in=C_in,
            C_out=C_latent,
            C_width=C_width,
            N_block=N_block,
            last_activation=None,
            outputs_at='global_mean',
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            with_gradient_features=with_gradient_features,
            with_gradient_rotations=with_gradient_rotations,
            diffusion_method='spectral',
        )

    def forward(self, x, mass, L, evals, evecs, gradX, gradY):
        """
        Args:
            x: (V, C_in) per-vertex input features
            mass, L, evals, evecs, gradX, gradY: geometric operators
        Returns:
            z: (C_latent,) global latent vector
        """
        return self.diffnet(x, mass, L, evals, evecs, gradX, gradY)


class DiffusionNetDecoder(nn.Module):
    """
    Decoder: global latent vector -> per-vertex features.

    Broadcasts the latent vector to all vertices, then applies
    DiffusionNet blocks to produce spatially-varying outputs.
    """

    def __init__(self, C_latent, C_out, C_width=64, N_block=3,
                 mlp_hidden_dims=None, dropout=False,
                 with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width]

        # Project latent to per-vertex initial features
        self.latent_to_vertex = nn.Linear(C_latent, C_width)

        self.diffnet = DiffusionNet(
            C_in=C_width,
            C_out=C_out,
            C_width=C_width,
            N_block=N_block,
            last_activation=None,
            outputs_at='vertices',
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            with_gradient_features=with_gradient_features,
            with_gradient_rotations=with_gradient_rotations,
            diffusion_method='spectral',
        )

    def forward(self, z, n_verts, mass, L, evals, evecs, gradX, gradY):
        """
        Args:
            z: (C_latent,) global latent vector
            n_verts: number of vertices to broadcast to
            mass, L, evals, evecs, gradX, gradY: geometric operators
        Returns:
            x_out: (V, C_out) per-vertex reconstructed features
        """
        # Broadcast latent to all vertices
        x = self.latent_to_vertex(z)       # (C_width,)
        x = x.unsqueeze(0).expand(n_verts, -1)  # (V, C_width)

        return self.diffnet(x, mass, L, evals, evecs, gradX, gradY)


class DiffusionNetAutoencoder(nn.Module):
    """
    Full autoencoder: HKS -> latent -> HKS reconstruction.

    The latent code is the shape embedding, which can be used for
    downstream clustering, visualization, correlation analysis, etc.
    """

    def __init__(self, C_in=16, C_latent=32, C_width=64, N_block_enc=3,
                 N_block_dec=3, mlp_hidden_dims=None, dropout=False,
                 with_gradient_features=True, with_gradient_rotations=True):
        """
        Args:
            C_in:       input feature dim (number of HKS scales)
            C_latent:   latent space dimension
            C_width:    internal DiffusionNet width
            N_block_enc: number of encoder blocks
            N_block_dec: number of decoder blocks
        """
        super().__init__()

        self.C_in = C_in
        self.C_latent = C_latent

        self.encoder = DiffusionNetEncoder(
            C_in=C_in,
            C_latent=C_latent,
            C_width=C_width,
            N_block=N_block_enc,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            with_gradient_features=with_gradient_features,
            with_gradient_rotations=with_gradient_rotations,
        )

        self.decoder = DiffusionNetDecoder(
            C_latent=C_latent,
            C_out=C_in,  # reconstruct HKS
            C_width=C_width,
            N_block=N_block_dec,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            with_gradient_features=with_gradient_features,
            with_gradient_rotations=with_gradient_rotations,
        )

    def encode(self, x, mass, L, evals, evecs, gradX, gradY):
        """Encode per-vertex features to a global latent vector."""
        return self.encoder(x, mass, L, evals, evecs, gradX, gradY)

    def decode(self, z, n_verts, mass, L, evals, evecs, gradX, gradY):
        """Decode a latent vector to per-vertex features."""
        return self.decoder(z, n_verts, mass, L, evals, evecs, gradX, gradY)

    def forward(self, x, mass, L, evals, evecs, gradX, gradY):
        """
        Full forward pass: encode then decode.
        Args:
            x: (V, C_in) per-vertex HKS features
        Returns:
            x_recon: (V, C_in) reconstructed HKS
            z: (C_latent,) latent code
        """
        z = self.encode(x, mass, L, evals, evecs, gradX, gradY)
        n_verts = x.shape[0]
        x_recon = self.decode(z, n_verts, mass, L, evals, evecs, gradX, gradY)
        return x_recon, z
