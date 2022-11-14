import cv2
import numpy as np

import random
from typing import List, Tuple
import math

import pkg_resources

from src.data._models import ImageDetails

def _check_image(size: Tuple[int, int], epsilon: float, ring_center: Tuple[int, int], brightness: Tuple[int, int]) -> None:
    width, height = size
    width_ring_center, height_ring_center = ring_center
    min_brightness, max_brightness = brightness
    _ = ImageDetails(width=width,
              height=height,
              epsilon=epsilon,
              ring_center_width=width_ring_center,
              ring_center_height=height_ring_center,
              min_brightness=min_brightness,
              max_brightness=max_brightness)

def compute_single_pixel(x: int,
                         y: int,
                         epsilon: float,
                         ring_center: Tuple[int, int],
                         diff_brightness: int,
                         diff_between_rings: float) -> float:
    """Compute color value of individual pixel

    :param x: X coordinate of a pixel
    :type x: int
    :param y: Y cooridante of a pixel
    :type y: int
    :param epsilon: Epsilon value
    :type epsilon: float
    :param ring_center: Position of central ring
    :type ring_center: Tuple[int, int]
    :param diff_brightness: Difference between maximal and minimal brightness
    :type diff_brightness: int
    :param diff_between_rings: Difference between rings
    :type diff_between_rings: float
    :return: Color value of a pixel
    :rtype: float
    """
    width_ring_center, height_ring_center = ring_center
    value = (diff_brightness * math.cos(2*math.pi*(1.0-epsilon + ((pow((x - width_ring_center)*2, 2) + pow((y - height_ring_center)*2, 2)) / diff_between_rings ))))
    return value

def generate_pure_image(size: Tuple[int, int],
                        epsilon: float, 
                        ring_center: Tuple[int, int], 
                        brightness: Tuple[int, int]) -> np.array:
    """Generate pure image

    :param size: Size of the image (width, height)
    :type size: Tuple[int, int]
    :param epsilon: Epsilon value
    :type epsilon: float
    :param ring_center: Position of the central ring
    :type ring_center: Tuple[int, int]
    :param brightness: Range of brightness
    :type brightness: Tuple[int, int]
    :return: 2D array which represents pure image
    :rtype: np.array
    """
    width, height = size
    min_brightness, max_brightness =  brightness
    
    mean_brightness = (min_brightness + max_brightness) / 2
    diff_brightness = max_brightness - mean_brightness
    
    diff_betweeen_rings_denominator = 6.07
    diff_between_rings = width*width / diff_betweeen_rings_denominator
    
    img=np.ones((height, width))
    img=img[:,:] * mean_brightness
    for x in range (width):
        for y in range(height):
            img[y,x]=img[y,x]+compute_single_pixel(x, y, epsilon, ring_center, diff_brightness, diff_between_rings)
    img=img.astype(np.uint8)
    return img

def load_random_noise_filename(seed: int=None) -> str:
    """Load filename of random noise from package sample dataset

    :return: Name of the .png file
    :rtype: str
    """
    random.seed(seed)
    noise_file_index = random.randint(0, 24)
    file = pkg_resources.resource_filename(__name__, f"/samples/noise/{noise_file_index}.png")
    return file

def add_noise_to_image(pure_image: np.array, noise: np.array) -> np.array:
    """Add random noise to the pure image

    :param pure_image: 2D array which represents pure image
    :type pure_image: np.array
    :param noise: 2D array which represents noise image
    :type noise: np.array
    :return: Noised image
    :rtype: np.array
    """
    noise = noise[:,:,0]
    noise_mean = np.mean(noise)
    difference = -(noise-noise_mean)
    noised_image = pure_image-difference
    noised_image = np.clip(noised_image, 0, 255)
    return noised_image

def generate_image(epsilon: float,
                   size: Tuple[int, int]=(640, 480),
                   ring_center: Tuple[int, int]=(320, 240),
                   brightness: Tuple[int, int]=(80, 210),
                   seed: int=None
                    ) -> np.array:
    """Generate the image

    :param epsilon: Epsilon value
    :type epsilon: float
    :param size: Size of the image (width, height), defaults to (640, 480)
    :type size: Tuple[int, int], optional
    :param ring_center: Position of central ring, defaults to (320, 240)
    :type ring_center: Tuple[int, int], optional
    :param brightness: Range of brightness, defaults to (80, 210)
    :type brightness: Tuple[int, int], optional
    :return: 2D array which represents an image
    :rtype: np.array
    """
    _check_image(size, epsilon, ring_center, brightness)
    
    
    pure_image = generate_pure_image(size, epsilon, ring_center, brightness)
    file = load_random_noise_filename(seed)
    random_noise_image = cv2.imread(file)
    
    if ( random_noise_image.shape[:2] != size):
        random_noise_image = cv2.resize(random_noise_image, size, interpolation=cv2.INTER_AREA)
    noised_image = add_noise_to_image(pure_image, random_noise_image)
    return noised_image.astype(np.uint8)

if __name__ == "__main__":
    noised_image = generate_image(epsilon=0.4)