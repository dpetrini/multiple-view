import numpy as np
from random import randint
import math
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def cropping(image, crop_size, dim1, dim2):
    """crop the image and pad it to in_size
    Args :
        images : numpy array of images
        crop_size(int) : size of cropped image
        dim1(int) : vertical location of crop
        dim2(int) : horizontal location of crop
    Return :
        cropped_img: numpy array of cropped image
    """
    cropped_img = image[dim1:dim1+crop_size, dim2:dim2+crop_size]
    return cropped_img


def add_elastic_transform(image, alpha, sigma, pad_size=30, seed=None):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor
        sigma :  σ is an elasticity coefficient
        random_state = random integer
        Return :
        image : elastically transformed numpy array of image
    """
    # image_size = int(image.shape[0])
    image = np.pad(image, pad_size, mode="symmetric")
    if seed is None:
        seed = randint(1, 100)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    # print(dx, dy)
    return cropping(map_coordinates(image, indices, order=1).reshape(shape),
                    512, pad_size, pad_size), seed


def flip(image, option_value):
    """
    Args:
        image : numpy array of image
        option_value = random integer between 0 to 3
    Return :
        image : numpy array of flipped image
    """
    if option_value == 0:
        # vertical
        image = np.flip(image, option_value)
    elif option_value == 1:
        # horizontal
        image = np.flip(image, option_value)
    elif option_value == 2:
        # horizontally and vertically flip
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    else:
        image = image
        # no effect
    return image


# https://github.com/jbohnslav/opencv_transforms
def _get_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute matrix for affine transformation
    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale  0]
    #                              [ sin(a)*scale    cos(a + shear)*scale   0]
    #                              [     0                  0               1]

    angle = math.radians(angle)
    shear = math.radians(shear)
    # scale = 1.0 / scale

    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
    RSS = np.array([[math.cos(angle)*scale, -math.sin(angle+shear)*scale, 0],
                   [math.sin(angle)*scale, math.cos(angle+shear)*scale, 0],
                   [0, 0, 1]])
    matrix = T @ C @ RSS @ np.linalg.inv(C)

    return matrix[:2, :]


def affine(img, angle, translate, scale, shear, interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_CONSTANT, fillcolor=0):
    """Apply affine transformation on the image keeping image center invariant
    Args:
        img (numpy ndarray): numpy ndarray to be transformed.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        interpolation (``cv2.INTER_NEAREST` or ``cv2.INTER_LINEAR`` or ``cv2.INTER_AREA``, ``cv2.INTER_CUBIC``):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, it is set to ``cv2.INTER_CUBIC``, for bicubic interpolation.
        mode (``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE`` or ``cv2.BORDER_REFLECT`` or ``cv2.BORDER_REFLECT_101``)
            Method for filling in border regions. 
            Defaults to cv2.BORDER_CONSTANT, meaning areas outside the image are filled with a value (val, default 0)
        val (int): Optional fill color for the area outside the transform in the output image. Default: 0
    """
    # if not _is_numpy_image(img):
    #     raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.shape[0:2]
    center = (img.shape[1] * 0.5 + 0.5, img.shape[0] * 0.5 + 0.5)
    matrix = _get_affine_matrix(center, angle, translate, scale, shear)

    # if img.shape[2]==1:
    #    return cv2.warpAffine(img, matrix, output_size[::-1],interpolation,
    #                          borderMode=mode, borderValue=fillcolor)[:,:,np.newaxis]
    # else:
    return cv2.warpAffine(img, matrix, output_size[::-1], interpolation,
                          borderMode=mode, borderValue=fillcolor)
