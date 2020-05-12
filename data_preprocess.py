import numpy as np
import cv2
from skimage import exposure
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm


def rotate_image(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def load_blur_img(path, img_size):
    img = cv2.imread(path)
    angle = np.random.randint(0, 360)
    img = rotate_image(img, angle)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, img_size)
    return img


def load_img_class(class_path, class_label, class_size, img_size, name):
    x = []
    y = []
    description = "Loading {} :".format(name)
    pbar = tqdm(range(class_size))
    pbar.set_description(name)
    for path in class_path:
        img = load_blur_img(path, img_size)
        x.append(img)
        y.append(class_label)
        pbar.update(1)

    while len(x) < class_size:
        rand_idx = np.random.randint(0, len(class_path))
        img = load_blur_img(class_path[rand_idx], img_size)
        x.append(img)
        y.append(class_label)
        pbar.update(1)

    return x, y


def load_data(img_size, class_size, hot_dogs, not_hot_dogs):
    img_size = (img_size, img_size)
    x_hot_dog, y_hot_dog = load_img_class(hot_dogs, 0, class_size, img_size, name="Hotdog")
    x_not_hot_dog, y_not_hot_dog = load_img_class(not_hot_dogs, 1, class_size, img_size, name="Not hotdog")
    print("There are", len(x_hot_dog), "hotdog images")
    print("There are", len(x_not_hot_dog), "not hotdog images")

    X = np.array(x_hot_dog + x_not_hot_dog)
    y = np.array(y_hot_dog + y_not_hot_dog)

    return X, y


def to_gray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html

    images = 0.2989 * images[:, :, :, 0] + 0.5870 * images[:, :, :, 1] + 0.1140 * images[:, :, :, 2]
    return images


def normalize_images(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)

    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])

    images = images.reshape(images.shape + (1,))
    return images


def preprocess_data(images):
    gray_images = to_gray(images)
    return normalize_images(gray_images)


def to_categorical(labels, num_classes):
    """
    One-hot-encode the labels
    """
    encoded_labels = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        encoded_labels[i, labels[i]] = 1
    return encoded_labels


def get_data():
    size = 32
    class_size = 20000
    # rand_state = 42
    # np.random.seed(rand_state)

    hotdogs = glob.glob('./input/seefood/train/hot_dog/**/*.jpg', recursive=True)
    hotdogs += glob.glob('./input/seefood/test/hot_dog/**/*.jpg', recursive=True)
    not_hotdogs = glob.glob('./input/seefood/train/not_hot_dog/**/*.jpg', recursive=True)
    not_hotdogs += glob.glob('./input/seefood/test/not_hot_dog/**/*.jpg', recursive=True)
    scaled_X, y = load_data(size, class_size, hotdogs, not_hotdogs)
    scaled_X = preprocess_data(scaled_X)
    scaled_X = scaled_X.swapaxes(1, -1)
    y = to_categorical(y, 2)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                        test_size=0.2,
                                                        random_state=None)
    return X_train, X_test, y_train, y_test
