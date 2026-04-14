# -*- coding: utf-8 -*-
"""
RST — Radio Spectrogram Transformer

Model based on DeiT (Data-efficient Image Transformer) adapted for
technosignature detection in radio observations.

Based on the AST (Audio Spectrogram Transformer) architecture by Yuan Gong et al.
Paper: https://arxiv.org/abs/2104.01778

Changes from AST:
- Input: (96, 1024) radio spectrogram at native resolution (no binning)
- Output: 1 class (sigmoid) for binary classification ETI vs RFI
- Patch stride: 16x16 (non-overlapping) -> 384 total patches
- No AudioSet pretraining (ImageNet only)
- Progressive unfreezing support via freeze/unfreeze layer methods
"""

import torch
import torch.nn as nn

import timm
try:
    from timm.layers import to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import to_2tuple, trunc_normal_


"""
PatchEmbed: projects 16x16 patches into the 768D embedding space.
Overrides timm implementation to accept arbitrary non-square input sizes.
"""
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, channels, height, width)
        # proj(x): (batch, 768, n_patch_h, n_patch_w)
        # flatten(2): (batch, 768, n_patch_h * n_patch_w)
        # transpose: (batch, num_patches, 768)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# --------------------------------------------------------------------------- #
#  RSTModel Main Architecture
# --------------------------------------------------------------------------- #
class RSTModel(nn.Module):
    """
    Radio Spectrogram Transformer for binary SETI classification.

    Args:
        label_dim: Output dimension. 1 for binary classification (sigmoid).
        fstride: Patch stride in the frequency dimension. 16 = non-overlapping.
        tstride: Patch stride in the time dimension. 16 = non-overlapping.
        input_fdim: Number of frequency bins in the input (1024 raw channels).
        input_tdim: Number of time bins (96 = 6 observations × 16 time bins).
        imagenet_pretrain: If True, load ImageNet pretrained weights.
        model_size: DeiT variant. Default 'base384' (~86M parameters).
        verbose: If True, print model information.
    """

    def __init__(
        self,
        label_dim: int = 1,
        fstride: int = 16,
        tstride: int = 16,
        input_fdim: int = 1024,
        input_tdim: int = 96,
        imagenet_pretrain: bool = True,
        model_size: str = 'base384',
        verbose: bool = True,
    ):
        super().__init__()

        if verbose:
            print('============== RST Model Summary ==============')
            print(f'ImageNet pretraining: {imagenet_pretrain}')
            print(f'Input shape: ({input_tdim}, {input_fdim})')
            print(f'Patch stride: ({fstride}, {tstride})')

        # Instantiate DeiT base model
        model_names = {
            'tiny224':  'deit_tiny_distilled_patch16_224',
            'small224': 'deit_small_distilled_patch16_224',
            'base224':  'deit_base_distilled_patch16_224',
            'base384':  'deit_base_distilled_patch16_384',
        }
        if model_size not in model_names:
            raise ValueError(f'model_size must be one of: {list(model_names.keys())}')

        self.v = timm.create_model(model_names[model_size], pretrained=imagenet_pretrain)

        # Save original pretrained patch embedding weights BEFORE replacing
        # the module, so we can use them for RGB→1 channel conversion later.
        if imagenet_pretrain:
            _orig_patch_weight = self.v.patch_embed.proj.weight.data.clone()
            _orig_patch_bias = self.v.patch_embed.proj.bias.data.clone()

        # Replace timm's PatchEmbed with our custom version that accepts arbitrary input sizes
        self.v.patch_embed = PatchEmbed(
            img_size=self.v.patch_embed.img_size if hasattr(self.v.patch_embed, 'img_size') else 384,
            patch_size=16,
            in_chans=3,
            embed_dim=self.v.embed_dim,
        )

        # Save the original dimensions of the pretrained DeiT model
        self.original_num_patches = self.v.patch_embed.num_patches      # e.g. 576 for 384x384
        self.original_hw = int(self.original_num_patches ** 0.5)        # e.g. 24 for 24x24
        self.original_embedding_dim = self.v.pos_embed.shape[2]         # 768 for base

        # Binary classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim),
        )

        # Wrapper to get the intemediate shape of the patch grid (f_dim, t_dim) based on the input size and stride
        f_dim, t_dim = self._get_patch_grid_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        if verbose:
            print(f'Frequency dim: {f_dim}, Time dim: {t_dim}')
            print(f'Number of patches: {num_patches}')

        # Adapt Patch Embedding from 3 channels (RGB) to 1 channel
        # Sums the 3 channels of the pretrained Conv2D to preserve weights.
        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        if imagenet_pretrain:
            # Sum the 3 RGB channels → 1 channel using the ORIGINAL weights
            new_proj.weight = nn.Parameter(
                torch.sum(_orig_patch_weight, dim=1).unsqueeze(1)
            )
            new_proj.bias = nn.Parameter(_orig_patch_bias)
        self.v.patch_embed.proj = new_proj

        # Adapt Positional Embedding to our patch grid size.
        # Strategy: Center crop for dims <= original (24), bilinear interpolate for dims > original.
        if imagenet_pretrain:
            # Take only the patch pos_embed (exclude CLS and DIST tokens)
            # Shape: (1, 576, 768) → reshape → (1, 768, 24, 24)
            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, self.original_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)
            )

            # Adapt the time dimension (dim 3, the "columns")
            # In the pos_embed 2D grid: dim 2 = freq (rows), dim 3 = time (cols)
            if t_dim <= self.original_hw:
                # CUT: take t_dim columns from the center of the 24×24 grid
                start = self.original_hw // 2 - t_dim // 2
                new_pos_embed = new_pos_embed[:, :, :, start:start + t_dim]
            else:
                # INTERPOLATE: stretch columns to fit more time patches
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(self.original_hw, t_dim), mode='bilinear'
                )

            # Adapt the frequency dimension (dim 2, the "rows")
            if f_dim <= self.original_hw:
                # CUT: take f_dim rows from the center
                start = self.original_hw // 2 - f_dim // 2
                new_pos_embed = new_pos_embed[:, :, start:start + f_dim, :]
            else:
                # INTERPOLATE: stretch rows to fit more frequency patches
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode='bilinear'
                )

            # Reshape back to (1, num_patches, 768)
            new_pos_embed = new_pos_embed.reshape(
                1, self.original_embedding_dim, num_patches
            ).transpose(1, 2)

            # Recombine with original CLS and DIST tokens
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
            )
        else:
            # Without pretraining: random positional embedding
            new_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 2, self.original_embedding_dim)
            )
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=0.02)

        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print(f'Total parameters: {total_params:,}')
            print('================================================')

    def _get_patch_grid_shape(self, fstride, tstride, input_fdim, input_tdim):
        """
        Compute how many patches there are in frequency (f_dim) and time (t_dim).

        Uses a dummy Conv2D to compute it automatically,
        so it works with any combination of input size and stride.
        """
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        """
        Forward pass of the RST model.

        Args:
            x: Input spectrogram, shape (batch, time_bins, freq_bins).
               Example: (32, 96, 1024)

        Returns:
            Raw logit, shape (batch, label_dim). To obtain the probability
            P(ETI), apply torch.sigmoid(output) during inference.
        """
        # (batch, 96, 1024) → add channel dimension → (batch, 1, 96, 1024)
        x = x.unsqueeze(1)
        # Swap freq and time: (batch, 1, 96, 1024) → (batch, 1, 1024, 96)
        # Because Conv2D expects (batch, channels, freq, time)
        x = x.transpose(2, 3)

        B = x.shape[0]

        # Patch embedding: (batch, 1, 1024, 96) → (batch, 384, 768)
        x = self.v.patch_embed(x)

        # Add special CLS and DIST tokens (inherited from DeiT)
        # Each token is a 768-number vector, expanded for the batch
        cls_tokens = self.v.cls_token.expand(B, -1, -1)   # (batch, 1, 768)
        dist_token = self.v.dist_token.expand(B, -1, -1)  # (batch, 1, 768)
        x = torch.cat((cls_tokens, dist_token, x), dim=1) # (batch, 386, 768)

        # Add positional embedding 
        x = x + self.v.pos_embed  # (batch, 386, 768)

        # Positional dropout (regularization)
        x = self.v.pos_drop(x)

        # Pass through the 12 Transformer blocks
        # Each block: LayerNorm → Multi-Head Attention → Residual → MLP → Residual
        for blk in self.v.blocks:
            x = blk(x)

        # Final layer normalization
        x = self.v.norm(x)

        # Take the average of CLS (index 0) and DIST (index 1) tokens
        # This 768-number vector is the "representation" of the cadence
        x = (x[:, 0] + x[:, 1]) / 2  # (batch, 768)

        # MLP head: LayerNorm(768) → Linear(768, 1) → raw logit
        x = self.mlp_head(x)  # (batch, 1)

        return x

    # Progressive Unfreezing Methods
    def freeze_backbone(self):
        """
        Phase 1: Freeze the entire Transformer, keep only the MLP head trainable.

        Useful in the first training phase: the backbone retains ImageNet
        weights while the head learns to classify ETI vs RFI.
        """
        for param in self.v.parameters():
            param.requires_grad = False
        for param in self.mlp_head.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 4):
        """
        Phase 2: Unfreeze the last N Transformer blocks + head.

        Args:
            n: Number of final blocks to unfreeze (default: 4).
               With 12 total blocks, n=4 unfreezes blocks 8, 9, 10, 11.
        """
        # First freeze everything
        self.freeze_backbone()
        # Then unfreeze the last n blocks
        for block in self.v.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        # Also unfreeze the final layer norm (connected to the blocks)
        for param in self.v.norm.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """
        Phase 3: Unfreeze the entire model for full fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params_count(self) -> int:
        """Count the number of currently trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
#  Quick test: verify the model works
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('--- Test 1: Forward pass with SETI input ---')
    model = RSTModel(
        label_dim=1,
        fstride=16,
        tstride=16,
        input_fdim=1024,
        input_tdim=96,
        imagenet_pretrain=True,
    )
    # Simulate a batch of 4 cadences, each (96 time, 1024 freq)
    test_input = torch.randn(4, 96, 1024)
    test_output = model(test_input)
    print(f'Input shape:  {test_input.shape}')   # (4, 96, 1024)
    print(f'Output shape: {test_output.shape}')   # (4, 1)

    print('\n--- Test 2: Progressive Unfreezing ---')
    model.freeze_backbone()
    print(f'Phase 1 (head only):    {model.get_trainable_params_count():,} trainable params')
    model.unfreeze_last_n_blocks(4)
    print(f'Phase 2 (last 4 blocks): {model.get_trainable_params_count():,} trainable params')
    model.unfreeze_all()
    print(f'Phase 3 (full):          {model.get_trainable_params_count():,} trainable params')
