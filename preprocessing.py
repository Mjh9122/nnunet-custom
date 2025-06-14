import numpy as np
import numpy.typing as npt

from typing import Union, Optional, Dict, List, Tuple, Any

NDArray = npt.NDArray[np.float32]
CT_CLIP_VALUES = (.5, 99.5)
MAX_3D_PATCH_SIZE = (256, 256, 256)
MIN_3D_BATCH_SIZE = 2
CROPPING_NORMALIZATION_THRESHOLD = .25

def load_data(filepath: str) -> NDArray:
    """Loads data from image file into numpy array for preprocessing

    Args:
        filepath (str): filepath to image

    Returns:
        NDArray: image
    """
    pass

def crop_zeros(image:NDArray) -> Tuple[NDArray, bool]:
    """Creates bounding box of nonzero values in an image. Also returns whether the total # of voxels 
    is reduced by more than 25%, triggering different normalization strategy later on. 

    Args:
        image (NDArray): original image

    Returns:
        Tuple[NDArray, bool]: cropped image, reduction > 25%
    """
    pass

def resample_image(image:NDArray, old_spacing:Tuple[float, float, float], new_spacing:Tuple[float, float, float]) -> NDArray:
    """Resamples image to new voxel spacing using cubic spline interpolation

    Args:
        image (NDArray): original image
        old_spacing (Tuple[float, float, float]): original voxel spacing (X, Y, Z)
        new_spacing (Tuple[float, float, float]): new voxel spacing (X, Y, Z)

    Returns:
        NDArray: resampled image
    """ 
    pass

def resample_mask(mask:NDArray, old_spacing:Tuple[float, float, float], new_spacing:Tuple[float, float, float]) -> NDArray:
    """Resamples mask to new voxel spacing using nearest neighbor interpolation

    Args:
        mask (NDArray): original mask
        old_spacing (Tuple[float, float, float]): original voxel spacing (X, Y, Z)
        new_spacing (Tuple[float, float, float]): new voxel spacing (X, Y, Z)

    Returns:
        NDArray: resampled mask
    """ 
    pass


def normalize(
        image:NDArray, 
        modality: str, 
        cropping_threshold_met: bool,
        dataset_stats: Optional[Tuple[float, float]] = None, 
        clipping_percentiles:Optional[Tuple[float, float]] = (.5, 99.5), 
        mask:Optional[NDArray] = None,
    ) -> NDArray:
    """Normalizes images using nnU-net normalization strategy. 

    For CT images: values are clipped using [.5 - 99.5]. percentile values of non-background values,
    followed by z-score normilization using dataset wide statistics. Dataset stats cannot be None when
    modality is 'ct'

    For other modalities: z-score normilization applied to each sample

    If cropping reduced number of voxels by > 25% then only nonzero values are used for normalization

    Args:
        image: original image
        modality: image modality ('ct', 'mri', etc) found via dataset JSON
        cropping_threshold_met: whether the cropping threshold for alternate normalization is met
        dataset_stats: (mean, std) dataset statistics for dataset wide CT normalization
        clipping_percentiles: (.5 - 99.5) preconfigured clipping percentages for CT normalization
        mask: Optional mask normalization within ROI
    Returns:
        NDArray: normalized image
    """
    pass

def compute_dataset_stats(
        dataset_dir: str, 
        modality: str, 
        image_suffix: str, 
        mask_suffix: str      
    ) -> Dict[str, Any]:
    """Calculate key statistics from datasets.
    From CT datasets: mean, std, .5, 99.5 values from foreground classes. Median shape and voxel spacing.
    From non-CT datasets: Median shape and voxel spacing. 

    Args:
        dataset_dir: path to dataset images
        modality: modality of images
        image_suffix: suffix to identify images in dir
        mask_suffix: suffix to identify masks in dir

    Returns:
        Dict[str, Tuple[float, float]]: 'percentiles':(low:float, high:float), 
                                        'stats':(mean:float, std:float),
                                        'shape':(int, int, int),
                                        'spacing':(float, float, float)

    """
    pass

def determine_cascade_necessity(median_shape: Tuple[float, float, float]) -> bool:
    """Determines if U-Net Cascade needed based on nnU-Net heuristics.
    
    Returns True if median shape contains >4x voxels than can be processed
    by 3D U-Net with patch size 128³ and batch size 2. Otherwise 3d cascade network is not trained.
    
    Args:
        median_shape (Tuple[float, float, float]): Median shape of post-resampled images

    Returns:
        bool: whether the cascade is determined to be necessary
    """

def lower_resolution(
        median_shape: Tuple[float, float, float], 
        voxel_spacing: Tuple[float, float, float], 
        image: NDArray
    ) -> Tuple[NDArray, Tuple[float, float, float]]:
    """Follows nnU-net algorithm for lowering image resolution for input into 3d cascade. 
    Determines appropriate size and voxel spacing by resampling images. 

    For anisotropic images, higher resolution axes are resampled first.
    Once resolution on all axis is the same, all axes are resampled simultaniously.

    Args:
        median_shape (Tuple[float, float, float]): Median shape of dataset
        voxel_spacing (Tuple[float, float, float]): voxel spacing of dataset
        image (NDArray): original image

    Returns:
        Tuple[NDArray, Tuple[float, float, float]]: low res image, new voxel spacing
    """

def modality_detection(json_path: str) -> str:
    """Detects image modality by searching json file for 'ct' substring to determine appropriate preprocessing steps

    Args:
        json_path (str): path of dataset's json file

    Returns:
        str: image modality detected
    """

class nnUNetPreprocessor:
    def __init__(self, dataset_config: Dict[str, Any]):
        """Create a nnUnet preprocessor

        Args:
            dataset_config (Dict[str, Any]): config dict with dataset dir, json, modality, etc.
        """
        self.config = dataset_config
        self.dataset_stats = None
    
    def preprocess_case(self, image_path: str, mask_path: str) -> Tuple[NDArray, NDArray]:
        """Preprocess a single image from start to finish

        Args:
            image_path (str): path to image
            mask_path (str): path to mask

        Returns:
            Tuple[NDArray, NDArray]: preprocessed image, preprocessed mask
        """