import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import math
from functools import partial

import torch
import torch.jit
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import Tensor

from collections import OrderedDict

from timm.layers import PatchDropout, trunc_normal_
from timm.models._manipulate import checkpoint_seq
import xformers.ops as xops




class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)



# CellViT implementation for segmentation evaluation
class CellViT(nn.Module):
    def __init__(self, encoder, encoder_dim=768, drop_rate=0.1):
        super().__init__()
        
        self.encoder = encoder

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.embed_dim = encoder_dim
        self.drop_rate = drop_rate

        # Set dimensions based on encoder size
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 256
        elif self.embed_dim < 1024:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        else:
            # Dimensions for ViT-Large and larger models
            self.skip_dim_11 = 768
            self.skip_dim_12 = 384
            self.bottleneck_dim = 768

        # Decoder architecture (simplified for brevity)
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        self.nuclei_binary_map_decoder = self.create_upsampling_branch(2)
        self.hv_map_decoder = self.create_upsampling_branch(2)

        self.initialize_weights()

    def initialize_weights(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)
        
        # Special initialization for Deconv2DBlock
        for module in self.modules():
            if isinstance(module, Deconv2DBlock):
                for layer in module.block:
                    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
        
        # Initialize the final layer of each decoder branch with smaller weights
        for branch in [self.nuclei_binary_map_decoder, self.hv_map_decoder]:
            final_conv = branch.decoder0_header[-1]
            if isinstance(final_conv, nn.Conv2d):
                nn.init.normal_(final_conv.weight, std=0.01)
                nn.init.constant_(final_conv.bias, 0)

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """
        Create Upsampling branch for segmentation
        """
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def check_tensor_for_nan(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=False)
            raise ValueError(f"NaN detected in {tensor_name} at indices: {nan_indices.tolist()}")

    def check_weights_for_nan(self):
        for name, param in self.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                nan_indices = torch.nonzero(torch.isnan(param), as_tuple=False)
                raise ValueError(f"NaN detected in trainable weights: {name} at indices: {nan_indices.tolist()}")

    def _forward_upsample(
        self,
        images: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        self.check_tensor_for_nan(images, "images in _forward_upsample")
        self.check_tensor_for_nan(f1, "f1 in _forward_upsample")
        self.check_tensor_for_nan(f2, "f2 in _forward_upsample")
        self.check_tensor_for_nan(f3, "f3 in _forward_upsample")
        self.check_tensor_for_nan(f4, "f4 in _forward_upsample")

        b4 = branch_decoder.bottleneck_upsampler(f4)
        self.check_tensor_for_nan(b4, "b4 after bottleneck_upsampler")

        b3 = self.decoder3(f3)
        self.check_tensor_for_nan(b3, "b3 after decoder3")

        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        self.check_tensor_for_nan(b3, "b3 after decoder3_upsampler")

        b2 = self.decoder2(f2)
        self.check_tensor_for_nan(b2, "b2 after decoder2")

        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        self.check_tensor_for_nan(b2, "b2 after decoder2_upsampler")

        b1 = self.decoder1(f1)
        self.check_tensor_for_nan(b1, "b1 after decoder1")

        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        self.check_tensor_for_nan(b1, "b1 after decoder1_upsampler")

        b0 = self.decoder0(images)
        self.check_tensor_for_nan(b0, "b0 after decoder0")

        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
        self.check_tensor_for_nan(branch_output, "branch_output")

        return branch_output

    def forward(self, images, magnification='40x'):
        # Check input images for NaN
        self.check_tensor_for_nan(images, "input images")

        # Check model weights for NaN
        self.check_weights_for_nan()

        out_dict = {}
        num_registers = 4  # Adjust based on actual model configuration
        features = self.encoder.get_intermediate_layers(images)
        f1, f2, f3, f4 = features
        
        # Determine the feature map size dynamically
        num_patches = f1.shape[1] - (num_registers+1)  # Subtract tokens
        feature_size = int(np.sqrt(num_patches))

        # Reshape features to [B, C, H, W] dynamically
        f1 = f1[:, (num_registers+1):, :].permute(0, 2, 1).view(f1.shape[0], -1, feature_size, feature_size)
        f2 = f2[:, (num_registers+1):, :].permute(0, 2, 1).view(f2.shape[0], -1, feature_size, feature_size)
        f3 = f3[:, (num_registers+1):, :].permute(0, 2, 1).view(f3.shape[0], -1, feature_size, feature_size)
        f4 = f4[:, (num_registers+1):, :].permute(0, 2, 1).view(f4.shape[0], -1, feature_size, feature_size)
        
        mask_logits = self._forward_upsample(
            images, f1, f2, f3, f4, self.nuclei_binary_map_decoder
        )
        out_dict["masks"] = F.log_softmax(mask_logits, dim=1).exp()
        out_dict["distances"] = self._forward_upsample(
                images, f1, f2, f3, f4, self.hv_map_decoder
        )

        return out_dict
    

class CellViTMultiClass(nn.Module):
    """
    CellViT variant for multi-class semantic segmentation.
    Single decoder branch with N output classes.
    """
    def __init__(self, encoder, encoder_dim=768, num_classes=22, drop_rate=0.1):
        super().__init__()
        
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.embed_dim = encoder_dim
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        # Set dimensions based on encoder size
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 256
        elif self.embed_dim < 1024:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        else:
            self.skip_dim_11 = 768
            self.skip_dim_12 = 384
            self.bottleneck_dim = 768

        # Shared decoder layers
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        # Single segmentation decoder for multi-class output
        self.segmentation_decoder = self.create_upsampling_branch(num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)
        
        # Initialize final layer with smaller weights for multi-class
        final_conv = self.segmentation_decoder.decoder0_header[-1]
        if isinstance(final_conv, nn.Conv2d):
            nn.init.normal_(final_conv.weight, std=0.01)
            nn.init.constant_(final_conv.bias, 0)

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create upsampling branch for multi-class segmentation."""
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def _forward_upsample(
        self,
        images: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        b4 = branch_decoder.bottleneck_upsampler(f4)
        b3 = self.decoder3(f3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        
        b2 = self.decoder2(f2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        
        b1 = self.decoder1(f1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        
        b0 = self.decoder0(images)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def forward(self, images, magnification='40x'):
        out_dict = {}
        num_registers = 4
        features = self.encoder.get_intermediate_layers(images)
        f1, f2, f3, f4 = features
        
        # Determine feature map size dynamically
        num_patches = f1.shape[1] - (num_registers + 1)
        feature_size = int(np.sqrt(num_patches))

        # Reshape features to [B, C, H, W]
        f1 = f1[:, (num_registers+1):, :].permute(0, 2, 1).view(f1.shape[0], -1, feature_size, feature_size)
        f2 = f2[:, (num_registers+1):, :].permute(0, 2, 1).view(f2.shape[0], -1, feature_size, feature_size)
        f3 = f3[:, (num_registers+1):, :].permute(0, 2, 1).view(f3.shape[0], -1, feature_size, feature_size)
        f4 = f4[:, (num_registers+1):, :].permute(0, 2, 1).view(f4.shape[0], -1, feature_size, feature_size)
        
        # Single decoder for multi-class segmentation
        seg_logits = self._forward_upsample(
            images, f1, f2, f3, f4, self.segmentation_decoder
        )
        
        out_dict["segmentation"] = seg_logits  # Raw logits

        return out_dict


########################################################
# Vision Transformer Implementation
########################################################


class PatchEmbed(nn.Module):
    """Convert image into patch embeddings with optional dual normalization."""
    def __init__(
        self, 
        img_size, 
        patch_size, 
        in_channels=3, 
        embed_dim=768, 
        dual_norm=False, 
        norm_layer=None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_patches = (img_size // patch_size)**2
        self.dual_norm = dual_norm
        
        self.project = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Initialize normalization layers if dual_norm is enabled
        if dual_norm and norm_layer is not None:
            self.pre_norm = norm_layer(in_channels)
            self.post_norm = norm_layer(embed_dim)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply pre-patch normalization if dual_norm is enabled
        if self.dual_norm:
            x_flat = x.flatten(2).transpose(1, 2)  # B, H*W, C
            x_flat = self.pre_norm(x_flat)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply patch embedding
        x = self.project(x)     # Batch X Embedding Dim X sqrt(N_patches) X sqrt(N_patches)
        x = x.flatten(2)        # Batch X Embedding Dim X N_patches
        x = x.transpose(1, 2)   # Batch X N_patches X Embedding Dim
        
        # Apply post-patch normalization if dual_norm is enabled
        if self.dual_norm:
            x = self.post_norm(x)
            
        return x


# Optimized SwiGLU implementation with xformers fallback
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)
        return self.w3(hidden)



# SwiGLU with optimized hidden dimension sizing
class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


    def forward(self, x):
        # Call parent class implementation
        hidden = super().forward(x)
        # Add dropout which might not be in the xformers implementation
        return hidden 


class TransformerBlock(nn.Module):
    """Transformer block with xformers attention and SwiGLU MLP."""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=None
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # QKV projection
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize QK normalization if enabled
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        # DropPath for stochastic depth
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        
        # Use SwiGLUFFNFused by default
        mlp_layer = mlp_layer or SwiGLUFFNFused
        
        # Calculate hidden features for MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # Create MLP
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=proj_drop,
        )
        
        # Layer scale parameters (if enabled)
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x):
        # Attention with xformers memory-efficient implementation
        shortcut = x
        x = self.norm1(x)
        
        # QKV projection
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Ensure consistent dtype
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # Use xformers memory-efficient attention
        x = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None,
            scale=self.scale
        )
        
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Apply first residual connection with optional layer scaling
        if self.gamma_1 is not None:
            x = shortcut + self.drop_path(self.gamma_1 * x)
        else:
            x = shortcut + self.drop_path(x)
        
        # MLP with residual connection and optional layer scaling
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class VisionTransformer(nn.Module):
    """
    Modern Vision Transformer implementation with:
    - XFormers memory-efficient attention
    - SwiGLU MLP activation
    - Dynamic position embeddings
    - Gradient checkpointing
    - Register tokens
    - Intermediate feature extraction
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        patch_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        weight_init="",
        norm_layer=None,
        act_layer=None,
        block_fn=TransformerBlock,
        mlp_layer=SwiGLUFFNFused,
        num_register_tokens=4,
    ):
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0 
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        self.numregisters = num_register_tokens

        # Patch embedding with optional dual normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
            dual_norm=dual_norm,
            norm_layer=norm_layer,
        )

        # Determine patch numbers
        if isinstance(img_size, tuple):
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            num_patches = (img_size // patch_size) ** 2

        # Token embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

        # Position embeddings
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        
        # Dropouts
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.patch_drop = PatchDropout(
            patch_drop_rate,
            num_prefix_tokens=self.num_prefix_tokens,
        ) if patch_drop_rate > 0 else nn.Identity()
        
        # Pre-norm
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        # Stochastic depth configuration
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]


        layer_init_values = []
        for i in range(depth):
            if depth < 18:
                layer_init_values.append(0.1)
            elif depth < 24:
                layer_init_values.append(1e-5)
            else:
                layer_init_values.append(1e-6)
        
        # Create transformer blocks
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=layer_init_values[i],
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()

        # Initialize weights
        if weight_init != "skip":
            self._init_weights()

    def _init_weights(self):
        """Initialize weights for the model"""
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)
        
        # Apply general weight init to all modules
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "register_tokens"}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def interpolate_pos_embed(self, x, h, w):
        """Efficiently interpolate position embeddings for different image sizes"""
        num_patches = x.shape[1] - 1  # excluding class token
        N = self.pos_embed.shape[1] - 1  # original number of patches
        
        if num_patches == N:
            return self.pos_embed
            
        # Extract class and patch position embeddings
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        # Calculate new dimensions
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        
        # Add small offset to avoid floating point issues
        w0, h0 = w0 + 0.1, h0 + 0.1
        
        # Interpolate patch position embeddings
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        # Combine class and patch position embeddings
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        """Efficiently prepare tokens with register tokens and position embeddings"""
        B, C, H, W = x.shape 
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1) if self.cls_token is not None else torch.zeros(B, 0, self.embed_dim, device=x.device)
        x = torch.cat((cls_token, x), dim=1) if self.cls_token is not None else x
        
        # Add position embeddings (with correct interpolation)
        pos_embed = self.interpolate_pos_embed(x, H, W)
        x = x + pos_embed
        
        # Add register tokens after class token and before patch tokens
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        if self.cls_token is not None:
            x = torch.cat((x[:, 0:1], reg_tokens, x[:, 1:]), dim=1)
        else:
            x = torch.cat((reg_tokens, x), dim=1)
        
        return self.pos_drop(x)

    def get_intermediate_layers(self, x):
        """Extract features at specified points in the network"""
        x = self.prepare_tokens(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        features = []
        total_blocks = len(self.blocks)
        extraction_points = [
            (total_blocks // 4) - 1, 
            (total_blocks // 2) - 1, 
            (3 * total_blocks // 4) - 1, 
            total_blocks - 1
        ]

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in extraction_points:
                features.append(x)

        return features
        
    def forward_features(self, x):
        """Forward pass through the features portion of the network"""
        x = self.prepare_tokens(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
            
        x = self.norm(x)
        return x

    def forward(self, x, return_type=None):
        """Forward pass with option to return different token types"""
        x = self.forward_features(x)
        
        if return_type is None:
            # Default return just the cls token if it exists, otherwise the first register token
            return x[:, 0] if self.cls_token is not None else x[:, 0]
        else:
            # Return structured output with different token types
            return {
                'cls_token': x[:, 0] if self.cls_token is not None else None,
                'register_tokens': x[:, 1:1+self.numregisters] if self.cls_token is not None else x[:, :self.numregisters],
                'patch_tokens': x[:, (1+self.numregisters):] if self.cls_token is not None else x[self.numregisters:]
            }

