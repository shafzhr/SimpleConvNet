import numpy as np
from skimage import exposure


def normalize_images(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)

    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])

    images = images.reshape(images.shape + (1,))
    return images


def to_gray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html

    images = 0.2989 * images[:, :, :, 0] + 0.5870 * images[:, :, :, 1] + 0.1140 * images[:, :, :, 2]
    return images


def preprocess_data(images):
    gray_images = to_gray(images)
    return normalize_images(gray_images)
