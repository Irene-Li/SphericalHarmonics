"""
DiffusionNet encoder → global binary classification of per-fate presence.

Architecture:
    DiffusionNet (HKS input, global_mean pooling)
    → fc_hidden (optional)
    → fc_out
    → (n_fates,) raw logits  [no sigmoid; use BCEWithLogitsLoss during training]

This is a deliberately minimal model: no latent space, no decoder, no KL term.
The goal is to verify that DiffusionNet can learn a meaningful global
representation of organoid morphology under a simpler, well-conditioned
classification objective, before returning to the full VAE.

The model outputs raw logits so that BCEWithLogitsLoss can be used directly
(numerically more stable than sigmoid + BCELoss). At inference time, apply
torch.sigmoid(logits) > 0.5 to obtain predicted binary labels.

Dead-neuron diagnosis: if the encoder collapses here too, the problem is
upstream of the loss (HKS normalisation, architecture width, learning rate).
If it works here but not in the VAE, the issue is the Chamfer + KL objective.
"""

import torch
import torch.nn as nn
from diffusion_net.layers import DiffusionNet


class FateCoverageNet(nn.Module):
    """
    DiffusionNet encoder for per-organoid cell-fate binary classification.

    Maps per-vertex HKS features → global mean pool → MLP head →
    per-fate raw logits (apply sigmoid > 0.5 at inference for yes/no labels).

    Args:
        C_in:       number of HKS time scales (input feature channels)
        n_fates:    number of fate targets (one binary label per cell-fate type)
        C_width:    DiffusionNet internal channel width
        N_block:    number of DiffusionNet blocks
        mlp_hidden: hidden dim in the prediction head (0 = linear head)
        dropout:    whether to use dropout inside DiffusionNet MiniMLPs
    """

    def __init__(self, C_in, n_fates, C_width=64, N_block=4,
                 mlp_hidden=64, dropout=False):
        super().__init__()
        self.C_in   = C_in
        self.n_fates = n_fates

        # DiffusionNet backbone with global mean pooling
        self.encoder = DiffusionNet(
            C_in=C_in,
            C_out=C_width,
            C_width=C_width,
            N_block=N_block,
            outputs_at='global_mean',
            dropout=dropout,
            diffusion_method='spectral',
        )

        # Prediction head
        if mlp_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(C_width, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, n_fates),
            )
        else:
            self.head = nn.Linear(C_width, n_fates)

    def forward(self, x_hks, mass, L, evals, evecs, gradX, gradY):
        """
        Args:
            x_hks:  (V, C_in)  per-vertex HKS features
            mass, L, evals, evecs, gradX, gradY: DiffusionNet geometric operators

        Returns:
            logits: (1, n_fates)  raw logits (before sigmoid).
                    Use BCEWithLogitsLoss during training.
                    At inference: torch.sigmoid(logits) > 0.5 → binary predictions.
        """
        h = self.encoder(x_hks, mass, L, evals, evecs, gradX, gradY)
        if h.dim() == 1:
            h = h.unsqueeze(0)   # ensure (1, C_width) for batch consistency
        logits = self.head(h)    # (1, n_fates) raw logits
        return logits
