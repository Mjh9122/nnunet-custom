import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from typing import Type, Union, Optional, Dict, List, Tuple, Any, Callable


def batch_wise_loss_func(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ Loss function described in nnU-net paper, summing cross entropy loss and dice loss.
    
    This loss function is used when the batch should be treated as a single psuedo-volume.
    When slices or small patches rather than whole patients appear in a sample. 
    Dice loss is computed over all voxels in the batch using custom implementation below
    Cross entropy loss is computed using PyTorch's implementation

    ** CE loss expects logits and integer class labels, Dice loss expects softmaxed and one hot encoded labels **

    Args:
        inputs (torch.Tensor): predicted class of voxels (B, C, H, W) or (B, C, D, H, W)
        targets (torch.Tensor): true class of voxels (B, H, W) or (B, D, H, W) with values {0, 1, ... (Num classes - 1)}

    Returns:
        torch.Tensor: Loss value
    """

def sample_wise_loss_func(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ Loss function described in nnU-net paper, summing cross entropy loss and dice loss.
    
    This loss function is used when the batch should be treated as a individual volumes.
    When close to full patients appear in each sample (3D unet with no cascade and first stage of cascade)
    Dice loss is computed using the implementation below for each sample and averaged over the batch
    Cross entropy loss is computed using PyTorch's implementation

    ** CE loss expects logits and integer class labels, Dice loss expects softmaxed and one hot encoded labels **
    
    Args:
        inputs (torch.Tensor): predicted class of voxels (B, C, H, W) or (B, C, D, H, W)
        targets (torch.Tensor): true class of voxels (B, H, W) or (B, D, H, W) with values {0, 1, ... (Num classes - 1)}

    Returns:
        torch.Tensor: Loss value
    """

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Implementation of multi-class adapted dice loss specified in equation 2 of the nnU-net paper. 
    Inputs should be post softmax application. 
    Targets should be one hot encoded.

    Args:
        inputs (torch.Tensor): predicted class of voxels (B, C, H, W) or (B, C, D, H, W) with softmax predictions
        targets (torch.Tensor): true class of voxels (B, C, H, W) or (B, C, D, H, W) with one hot encoded classes in each C. 

    Returns:
        torch.Tensor: loss
    """

def epoch(
        model:nn.Module, 
        train_dataloader:DataLoader,
        val_dataloader:DataLoader,
        augmentation_func:Callable[[torch.Tensor], torch.Tensor], 
        loss_func:Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        optimizer:optim.Optimizer,
        num_batches:int = 250,
    )-> Tuple[float, float]:
    """Trains model for one epoch.

    Loads preprocessed training data from dataloader, applies data augmentation techniques, applies
    model and takes optimization steps. Average training and valiation losses are returned for LR sheduling 
    and early stopping logic. 

    Optimizer should be Adam with the appropriate LR based on sheduling.
    Orignal LR: 3 x 10^-4
    LR reduced by 5x when exponential moving average of training loss does not improve by at least 5 x 10^-3 
    over 30 epochs. 

    Training is stopped early when exponential moving average of valication loss does not improve by at least 
    5 x 10^-3 over 60 epochs, and LR < 10^-6

    Validation is performed using the same loss function without augmentation.
    One epoch is 250 batches by default.

    Args:
        model (nn.Module): Unet to be trained
        train_dataloader (DataLoader): Preprocessed training data 
        val_dataloader (DataLoader): Preprocessed validation data
        augmentation_func (Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]): 
            Random augmentation function that takes (image, mask) and returns (augmented_image, augmented_mask)
        loss_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function
        optimizer (optim.Optimizer): Optimizer
        num_batches (int, optional): Number of batches in one Epoch. Defaults to 250.

    Returns:
        Tuple[float, float]: (training loss, validation loss)
    """
