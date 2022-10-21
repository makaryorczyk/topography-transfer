from PIL import Image
import cv2
import numpy as np
import random
from tqdm import tqdm

import os

from generate_dataset import save2directory

ABSOLUTE_PATH = os.path.dirname("topography-transfer")

NOISE_PATH = ABSOLUTE_PATH+"data/interim/generated/noise/"
RAW_FRAMES_PATH = ABSOLUTE_PATH+"data/raw/frames/"

def _check_args(num_images: int, num_used_raw_image: int):
    if num_images <= 0:
        raise ValueError("Number of generated images must be grater than 0.")
    if num_used_raw_image <= 20:
        raise ValueError("Number of images used to extract noise must be greater than 20.")
    

def generate_random_noise(num_used_raw_images: int=100) -> np.array:
    """Generate random noise image which will be added to artificially generated pure images.
    Method randomly takes num_images raw frames and extracts noise from them.

    :param num_used_raw_images: Number of images that are used to create one noise image, defaults to 100
    :type num_used_raw_images: int, optional
    """
    noise_image = np.zeros((480,640))
    for _ in range(num_used_raw_images):
        img = cv2.imread(RAW_FRAMES_PATH+random.choice(os.listdir(RAW_FRAMES_PATH)))
        img = img[4:484,4:644,0]
        noise_image = (noise_image+img)
    noise_image = noise_image/num_used_raw_images
    return noise_image.astype(np.uint8)


def generate_noise_dataset(num_images: int=50, num_used_raw_images: int=100) -> None:
    """Generate random noise images

    :param num_images: Number of the generated noise images, defaults to 50
    :type num_images: int, optional
    :param num_used_raw_images: Number of images that are used to create one noise image, defaults to 100
    :type num_used_raw_images: int, optional
    """
    _check_args(num_images, num_used_raw_images)
    for frame in tqdm(range(num_images)):
        noise_image = generate_random_noise(num_used_raw_images)
        save2directory(noise_image, img_filename=f"{frame}.png", path=NOISE_PATH)




if __name__ == "__main__":
    pass