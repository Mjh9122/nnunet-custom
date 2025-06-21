import SimpleITK as sitk
import numpy as np

arr = np.ones((2, 2), np.float32)
img = sitk.GetImageFromArray(arr)
sitk.WriteImage(img, "test002.nii.gz")
