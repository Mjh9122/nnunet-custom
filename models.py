import torch 
import torch.nn as nn
import torch.functional as F

from typing import Union, Optional, Dict, List, Tuple, Any

class Unet2d(nn.Module):
    def __init__(self, channels: List[int], pooling_layers: List[int]) -> None:
        """Implementation of 2D Unet as described by the original Unet paper. 
        Included in nnU-net as 2D models are believed to perform better on some anisotropic datasets

        Implementaton Details:
        Uses leaky ReLU w/ slope 1e-2
        Uses instance normalization rather than batch
        Does not include residual connections populatized after orig. nnU-net paper
        Each pooling layer contains 2 3x3 conv layers 
        Upconv layers are transposed conv ops
        Skip connections connect encoder layer to corrisponding decoder layers

        Args:
            channels (List[int]): number of channels after each pooling layer
            pooling_layers (List[int]): number of pooling layers on each axis
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 2D Unet 

        Args:
            x (torch.Tensor): Input patch

        Returns:
            torch.Tensor: Logits
        """

class Unet3d(nn.Module):
    def __init__(self, channels: List[int], pooling_layers: List[int]) -> None:
        """Implementation of 3D Unet as described by the original 3D Unet paper. 2D operations are
        swapped for their 3D counterparts.

        Implementaton Details:
        Uses leaky ReLU w/ slope 1e-2
        Uses instance normalization rather than batch
        Does not include residual connections populatized after orig. nnU-net paper
        Each pooling layer contains 2 3x3x3 conv layers 
        Upconv layers are transposed conv ops
        Skip connections connect encoder layer to corrisponding decoder layers

        Args:
            channels (List[int]): number of channels after each pooling layer
            pooling_layers (List[int]): number of pooling layers on each axis
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 3D Unet 

        Args:
            x (torch.Tensor): Input patch

        Returns:
            torch.Tensor: Logits
        """

class CascadeUnet3d(nn.Module):
    def __init__(
            self, 
            low_res_channels: List[int], 
            low_res_pooling_layers: List[int],
            high_res_channels: List[int], 
            high_res_pooling_layers: List[int],
        ) -> None:
        """Implementation of 3D Cascade Unet as described by the original Unet paper. 
        
        Implementaton Details:
        Stage 1 (low res) Unet creates predictions on downsampled image
        Predictions are upsampled and combined with original image for refinement
        Stage 2 (high res) Unet refines predictions to original resolution
        Uses leaky ReLU w/ slope 1e-2
        Uses instance normalization rather than batch
        Does not include residual connections populatized after orig. nnU-net paper
        Each pooling layer contains 2 3x3x3 conv layers 
        Upconv layers are transposed conv ops
        Skip connections connect encoder layer to corrisponding decoder layers


        Args:
            low_res_channels: number of channels per pooling layer in low res unet
            low_res_pooling_layers: number of pools per axis in low res unet
            high_res_channels: number of channels per pooling layer in high res unet
            high_res_pooling_layers: number of pools per axis in high res unet
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 3D Cascade Unet 

        Args:
            x (torch.Tensor): Input patch

        Returns:
            torch.Tensor: Logits
        """
