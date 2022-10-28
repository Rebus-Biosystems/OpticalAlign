import os
import operator
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple

# TODO: make object oriented with imagestack class (see: starfish)


def read_lsgd_as_imagestack(filepath: str, image_ysize: int, image_xsize: int = -1, num_images: int = 12) -> np.ndarray:
    """
    Read binary data in .lsgd and return as a 3-D numpy array (num_images, y, x).

    Parameters
    ----------
    filepath : str
        filepath of the .lsgd file
    image_ysize : int
        pixel height of the image (eg. 2160 or 2048)
    image_xsize : int
        pixel width of the image (eg. 2560 or 2048)
        (default: -1) will infer the width automatically
    num_images : int
        depth of image stack (ie. 12)
    """

    # read binary data
    # reference: https://stackoverflow.com/questions/2146031/what-is-the-equivalent-of-fread-from-matlab-in-python
    with open(filepath, "rb") as fid:
        lsgd_imagestack = np.fromfile(fid, np.uint16)
        lsgd_imagestack = lsgd_imagestack.reshape((num_images, image_ysize, image_xsize))
        # shape = lsgd_imagestack.shape

    # check image shape matches expected shape
    # shape = lsgd_imagestack.shape
    # assert shape == (12, 2048, 2048) or shape == (12, 2160, 2560), "Image size does not match expected size!"

    return lsgd_imagestack


def imagestack_to_simage(imagestack: np.ndarray) -> np.ndarray:
    """
    Compute 2-D S image from 3-D imagestack consisting of 12 images.

    Parameters
    ----------
    imagestack : np.ndarray
        12 x m x n imagestack from SAO imaging .lsgd
    """

    # check there are 12 images
    # assert imagestack.shape[0] == 12, "Imagestack must have 12 images!"

    first_three_images = np.max(imagestack[0:3, :, :], axis=0)
    third_three_images = np.max(imagestack[6:9, :, :], axis=0)
    s_image = (first_three_images + third_three_images) / 2

    # check the image size is correct (flattened imagestack)
    # assert s_image.shape == imagestack.shape[1:3], "Image size was changed!"

    return s_image.round().astype("uint16")  # S images are uint16


def imagestack_to_modimage(
    imagestack: np.ndarray, kernel_size: int = 3, sigma: float = 0.2, gamma: float = 1.0
) -> np.ndarray:
    """
    Compute 2-D Mod image from 3-D imagestack consisting of 12 images.

    Parameters
    ----------
    imagestack : np.ndarray
        12 x m x n imagestack from SAO imaging .lsgd
    kernel_size : Tuple[int, int]
        size of Gaussian kernel. (default: (3, 3))
    sigma : float
        sigma of Gaussian kernel. (default: 0.2)
    gamma : float
        gamma order of returned matrix. (default: 1.0)
    """

    # check there are 12 images
    assert imagestack.shape[0] == 12, "Imagestack must have 12 images!"

    # use truncate to control kernel width in gaussian_filter
    trunc = (((kernel_size - 1) / 2) - 0.5) / sigma
    # gaussian filter
    imagestack = gaussian_filter(imagestack.astype("<f8"), sigma=(0, sigma, sigma), truncate=trunc)

    # calculate coefficient of variance for each "channel" and gamma correct
    first_three_images = (
        np.std(imagestack[0:3, :, :], axis=0, ddof=1) / np.mean(imagestack[0:3, :, :], axis=0)
    ) ** gamma
    second_three_images = (
        np.std(imagestack[3:6, :, :], axis=0, ddof=1) / np.mean(imagestack[3:6, :, :], axis=0)
    ) ** gamma
    third_three_images = (
        np.std(imagestack[6:9, :, :], axis=0, ddof=1) / np.mean(imagestack[6:9, :, :], axis=0)
    ) ** gamma
    fourth_three_images = (
        np.std(imagestack[9:12, :, :], axis=0, ddof=1) / np.mean(imagestack[9:12, :, :], axis=0)
    ) ** gamma

    mod_image = first_three_images * second_three_images * third_three_images * fourth_three_images

    # check the image size is correct (flattened imagestack)
    assert mod_image.shape == imagestack.shape[1:3], "Image size was changed!"

    return mod_image.clip(min=0, max=1)


def crop_center(imagestack: np.ndarray, crop_size: int = 1366) -> np.ndarray:
    """
    Crop center of 3D or 2D image to square with x and y size of crop_size.

    Parameters
    ----------
    imagestack : np.ndarray
        (12 x m x n) or (m x n) numpy array
    crop_size : int
        x and y dimensions of square cropped image (default: 1366x1366)
    """

    if imagestack.ndim == 3:
        imagestack = imagestack.transpose(1, 2, 0)  # switch to (m x n x 12)
    bounding = (crop_size, crop_size)
    start = tuple(map(lambda a, da: a // 2 - da // 2, imagestack.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))

    cropped_stack = imagestack[slices]

    if cropped_stack.ndim == 3:
        cropped_stack = cropped_stack.transpose(2, 0, 1)  # switch back to (12 x m x n)

    return cropped_stack
