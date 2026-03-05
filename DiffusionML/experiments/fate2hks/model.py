"""
DiffusionNet model for cross-prediction between HKS and cell fate markers.

Unlike the autoencoder in model.py (which uses global pooling for a shape
embedding), this model operates entirely at the per-vertex level: it takes
per-vertex input features and predicts per-vertex output features through
DiffusionNet blocks.
"""

import torch
import torch.nn as nn

from diffusion_net.layers import DiffusionNet


class DiffusionNetPredictor(nn.Module):
    """
    Per-vertex feature predictor using DiffusionNet.

    Maps (V, C_in) -> (V, C_out) through DiffusionNet blocks,
    enabling cross-prediction between different per-vertex feature
    spaces (e.g. HKS -> fate markers, or fate markers -> HKS).
    """

    def __init__(self, C_in, C_out, C_width=64, N_block=4,
                 mlp_hidden_dims=None, dropout=False,
                 with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width]

        self.C_in = C_in
        self.C_out = C_out

        self.diffnet = DiffusionNet(
            C_in=C_in,
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

    def forward(self, x, mass, L, evals, evecs, gradX, gradY):
        """
        Args:
            x: (V, C_in) per-vertex input features
            mass, L, evals, evecs, gradX, gradY: geometric operators
        Returns:
            y: (V, C_out) per-vertex predicted features
        """
        return self.diffnet(x, mass, L, evals, evecs, gradX, gradY)
