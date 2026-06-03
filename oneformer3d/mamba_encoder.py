"""SparseMambaEncoder: U-Net backbone with bidirectional Mamba SSM blocks.

Replaces SpConvUNet's residual-only encoder with Mamba SSM layers inserted
after the residual blocks at each level.  Voxels are sorted into horizontal
"slabs" (Z // slab_thickness) and scanned in both forward and backward
directions so every voxel accumulates full-scene context.
"""
import functools
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS
from .spconv_unet import ResidualBlock


class SparseMambaBlock(nn.Module):
    """Apply Mamba SSM to sparse voxel features using slab-based ordering.

    Voxels are sorted by (batch, z_slab, y, x) – where z_slab = z //
    slab_thickness – to create a spatially coherent 1-D sequence for Mamba.
    Each sample in a batch is processed independently so that the SSM state
    does not bleed across scenes.

    Args:
        channels (int): Feature dimension of the sparse tensor.
        d_state (int): Mamba SSM hidden state size.
        d_conv (int): Mamba depthwise convolution kernel width.
        expand (int): Mamba inner expansion factor.
        slab_thickness (int): Number of voxel rows (Z-axis) per slab.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        slab_thickness: int = 5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.slab_thickness = slab_thickness
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(channels)

        from mamba_ssm import Mamba
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Separate backward Mamba so each direction specialises independently.
        # A shared Mamba (batch-of-2 trick) is ~7% faster but causes noisy loss
        # because top→crown and crown→top gradients conflict in one parameter set.
        if bidirectional:
            self.mamba_bwd = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _slab_order(self, indices: torch.Tensor):
        """Return sort permutation and its inverse for slab-based ordering.

        Lexicographic key: batch (most significant) → z_slab → y → x.
        Stable sort is used at each level so ties in one key preserve the
        order established by the next (less significant) key.

        Args:
            indices (Tensor): ``(N, 4)`` integer tensor ``[batch, z, y, x]``.

        Returns:
            Tuple[Tensor, Tensor]: ``order`` and ``restore`` permutations,
            each of shape ``(N,)``.
        """
        b    = indices[:, 0]
        z    = indices[:, 1]
        y    = indices[:, 2]
        x    = indices[:, 3]
        slab = z // self.slab_thickness

        # Sort least-significant → most-significant (stable so later keys
        # dominate without clobbering finer structure).
        order = torch.argsort(x,       stable=True)
        order = order[torch.argsort(y   [order], stable=True)]
        order = order[torch.argsort(slab[order], stable=True)]
        order = order[torch.argsort(b   [order], stable=True)]

        restore = torch.argsort(order, stable=True)
        return order, restore

    # ------------------------------------------------------------------

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Forward pass.

        Args:
            x (SparseConvTensor): Input sparse tensor with features
                ``(N, channels)``.

        Returns:
            SparseConvTensor: Tensor with Mamba-updated features, same
            sparsity pattern as the input.
        """
        features = x.features  # (N, C)
        indices  = x.indices   # (N, 4)  [batch, z, y, x]

        order, restore = self._slab_order(indices)

        # Re-order voxels so that each contiguous chunk belongs to one batch.
        sorted_feats    = features[order]          # (N, C)
        sorted_batch    = indices[order, 0]        # (N,)

        out_feats = torch.empty_like(sorted_feats)

        ptr = 0
        for b_id in sorted_batch.unique(sorted=True):
            n = (sorted_batch == b_id).sum().item()
            feats_b = sorted_feats[ptr : ptr + n]      # (N_b, C)

            residual  = feats_b
            normed    = self.norm(feats_b).unsqueeze(0)  # (1, N_b, C)
            fwd       = self.mamba(normed).squeeze(0)    # (N_b, C)
            if self.bidirectional:
                bwd     = self.mamba_bwd(normed.flip(1)).squeeze(0).flip(0)
                feats_b = fwd + bwd + residual
            else:
                feats_b = fwd + residual

            out_feats[ptr : ptr + n] = feats_b
            ptr += n
        
        del features, indices, sorted_feats, sorted_batch
        return x.replace_feature(out_feats[restore])


# ---------------------------------------------------------------------------


@MODELS.register_module()
class SparseMambaEncoder(nn.Module):
    """Sparse U-Net backbone augmented with Mamba SSM sequence modeling.

    Mirrors the recursive structure of :class:`SpConvUNet` but inserts a
    :class:`SparseMambaBlock` after the residual blocks at every encoder
    level.  This allows the network to capture long-range spatial context
    while retaining the multi-scale skip-connection design of the original.

    The forward signature is identical to ``SpConvUNet`` so the module can
    be used as a drop-in replacement (set ``type='SparseMambaEncoder'`` in
    the config).

    Args:
        num_planes (List[int]): Feature channels at each U-Net level, from
            shallowest to deepest (e.g. ``[32, 64, 96, 128, 160]``).
        norm_fn (Callable): Normalization layer constructor.
        block_reps (int): Number of residual blocks per level.
        block (Callable | str): Block class (or ``'residual'``).
        indice_key_id (int): SpConv indice key counter for this level.
        normalize_before (bool): Whether to apply norm before conv.
        return_blocks (bool): If ``True``, also return the list of per-level
            encoder outputs (needed by the decoder for skip connections).
        d_state (int): Mamba SSM hidden-state dimension.
        d_conv (int): Mamba depthwise convolution kernel width.
        expand (int): Mamba inner expansion factor.
        slab_thickness (int): Voxel height (Z-axis) of each ordering slab.
    """

    def __init__(
        self,
        num_planes,
        norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
        block_reps: int = 2,
        block=ResidualBlock,
        indice_key_id: int = 1,
        normalize_before: bool = True,
        return_blocks: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        slab_thickness: int = 5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        if isinstance(block, str):
            assert block == 'residual', f'Unsupported block type: {block!r}'
            block = ResidualBlock

        # ── encoder residual blocks ──────────────────────────────────────
        enc_blocks = OrderedDict({
            f'block{i}': block(
                num_planes[0],
                num_planes[0],
                norm_fn,
                normalize_before=normalize_before,
                indice_key=f'subm{indice_key_id}',
            )
            for i in range(block_reps)
        })
        self.blocks = spconv.SparseSequential(enc_blocks)

        # ── Mamba SSM block ──────────────────────────────────────────────
        self.mamba_block = SparseMambaBlock(
            channels=num_planes[0],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            slab_thickness=slab_thickness,
            bidirectional=bidirectional,
        )

        # ── deeper levels (recursive) ────────────────────────────────────
        if len(num_planes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(num_planes[0]),
                    nn.ReLU(),
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}',
                    ),
                )
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}',
                    ),
                    norm_fn(num_planes[1]),
                    nn.ReLU(),
                )

            self.u = SparseMambaEncoder(
                num_planes[1:],
                norm_fn,
                block_reps,
                block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                slab_thickness=slab_thickness,
                bidirectional=bidirectional,
            )

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(num_planes[1]),
                    nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}',
                    ),
                )
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}',
                    ),
                    norm_fn(num_planes[0]),
                    nn.ReLU(),
                )

            # tail blocks merge skip connection + upsampled features
            tail_blocks = OrderedDict({
                f'block{i}': block(
                    num_planes[0] * (2 - i),
                    num_planes[0],
                    norm_fn,
                    indice_key=f'subm{indice_key_id}',
                    normalize_before=normalize_before,
                )
                for i in range(block_reps)
            })
            self.blocks_tail = spconv.SparseSequential(tail_blocks)

    # ------------------------------------------------------------------

    def forward(self, input, previous_outputs=None):
        """Forward pass.

        Args:
            input (SparseConvTensor): Sparse input at the current scale.
            previous_outputs (List[SparseConvTensor] | None): Accumulated
                encoder outputs from shallower levels (only used when
                ``return_blocks=True``).

        Returns:
            SparseConvTensor: Decoded output at the current scale.
            List[SparseConvTensor] (only when ``return_blocks=True``):
                Encoder block outputs collected so far.
        """
        # ── local feature extraction + Mamba context ────────────────────
        output = self.blocks(input)
        output = self.mamba_block(output)

        identity = spconv.SparseConvTensor(
            output.features,
            output.indices,
            output.spatial_shape,
            output.batch_size,
        )

        # ── encoder-decoder recursion ────────────────────────────────────
        if len(self.num_planes) > 1:
            output_decoder = self.conv(output)

            if self.return_blocks:
                output_decoder, previous_outputs = self.u(
                    output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)

            output_decoder = self.deconv(output_decoder)

            # skip connection: concat encoder identity + upsampled decoder
            output = output.replace_feature(
                torch.cat(
                    (identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        # ── collect encoder outputs for skip connections ─────────────────
        if self.return_blocks:
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs

        return output
