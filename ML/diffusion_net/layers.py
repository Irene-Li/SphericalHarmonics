"""
DiffusionNet layers.
Adapted from https://github.com/nmwsharp/diffusion-net (Sharp et al., 2022).
"""

import numpy as np
import torch
import torch.nn as nn

from .utils import toNP
from .geometry import to_basis, from_basis


class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.
    In the spectral domain: f_out = exp(lambda_i * t) * f_in
    """

    def __init__(self, C_inout, method='spectral'):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))
        self.method = method
        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, x, L, mass, evals, evecs):
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim should be C = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':
            x_spec = to_basis(x, evecs, mass)
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * self.diffusion_time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec
            x_diffuse = from_basis(x_diffuse_spec, evecs)

        elif self.method == 'implicit_dense':
            V = x.shape[-2]
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)
        else:
            raise ValueError("unrecognized method: " + self.method)

        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors using a learned complex-linear layer.
    Input:  vectors (V,C,2)
    Output: dots (V,C)
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()
        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

    def forward(self, vectors):
        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectors[..., 0] * vectorsBreal + vectors[..., 1] * vectorsBimag
        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    """A simple MLP with configurable hidden layer sizes."""

    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()
        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5))

            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation())


class DiffusionNetBlock(nn.Module):
    """A single DiffusionNet block: diffusion + gradient features + MLP + skip connection."""

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True,
                 diffusion_method='spectral',
                 with_gradient_features=True,
                 with_gradient_rotations=True):
        super(DiffusionNetBlock, self).__init__()

        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)

        self.MLP_C = 2 * self.C_width  # x_in + x_diffuse

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width

        self.mlp = MiniMLP(
            [self.MLP_C] + self.mlp_hidden_dims + [self.C_width],
            dropout=self.dropout)

    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):
        B = x_in.shape[0]
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim should be C = {}".format(
                    x_in.shape, self.C_width))

        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        if self.with_gradient_features:
            x_grads = []
            for b in range(B):
                x_gradX = torch.mm(gradX[b, ...], x_diffuse[b, ...])
                x_gradY = torch.mm(gradY[b, ...], x_diffuse[b, ...])
                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)
            x_grad_features = self.gradient_features(x_grad)
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        x0_out = self.mlp(feature_combined)
        x0_out = x0_out + x_in  # skip connection
        return x0_out


class DiffusionNet(nn.Module):
    """
    The full DiffusionNet architecture.

    Parameters:
        C_in:           input feature dimension
        C_out:          output feature dimension
        C_width:        internal block width (default: 128)
        N_block:        number of DiffusionNet blocks (default: 4)
        last_activation: optional activation on output (default: None)
        outputs_at:     'vertices', 'edges', 'faces', or 'global_mean'
        mlp_hidden_dims: hidden dims for per-block MLPs
        dropout:        use dropout in MLPs
        with_gradient_features:   use spatial gradient features
        with_gradient_rotations:  learn gradient rotations
        diffusion_method:         'spectral' or 'implicit_dense'
    """

    def __init__(self, C_in, C_out, C_width=128, N_block=4,
                 last_activation=None, outputs_at='vertices',
                 mlp_hidden_dims=None, dropout=True,
                 with_gradient_features=True, with_gradient_rotations=True,
                 diffusion_method='spectral'):
        super(DiffusionNet, self).__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.last_activation = last_activation
        self.outputs_at = outputs_at

        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']:
            raise ValueError("invalid outputs_at: " + outputs_at)

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        self.diffusion_method = diffusion_method
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # First and last linear layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        # DiffusionNet blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.N_block):
            block = DiffusionNetBlock(
                C_width=C_width,
                mlp_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations)
            self.blocks.append(block)

    def forward(self, x_in, mass, L=None, evals=None, evecs=None,
                gradX=None, gradY=None, edges=None, faces=None):
        """
        Forward pass.
        x_in:  (N,C_in) or (B,N,C_in)
        mass:  (N,) or (B,N)
        Returns: (N,C_out) or (B,N,C_out) depending on outputs_at
        """
        if x_in.shape[-1] != self.C_in:
            raise ValueError(
                "DiffusionNet C_in={}, but x_in last dim={}".format(
                    self.C_in, x_in.shape[-1]))

        # Add batch dim if not present
        if len(x_in.shape) == 2:
            appended_batch_dim = True
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L is not None: L = L.unsqueeze(0)
            if evals is not None: evals = evals.unsqueeze(0)
            if evecs is not None: evecs = evecs.unsqueeze(0)
            if gradX is not None: gradX = gradX.unsqueeze(0)
            if gradY is not None: gradY = gradY.unsqueeze(0)
            if edges is not None: edges = edges.unsqueeze(0)
            if faces is not None: faces = faces.unsqueeze(0)
        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        else:
            raise ValueError("x_in should be [N,C] or [B,N,C]")

        # First linear
        x = self.first_lin(x_in)

        # Blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)

        # Last linear
        x = self.last_lin(x)

        # Output remapping
        if self.outputs_at == 'vertices':
            x_out = x
        elif self.outputs_at == 'edges':
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        elif self.outputs_at == 'faces':
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        elif self.outputs_at == 'global_mean':
            # Area-weighted mean (discretization-invariant)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)

        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out
