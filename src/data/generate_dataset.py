from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from zipfile import ZipFile
import random
from tqdm import tqdm
import os


from generate_image import generate_image
from _models import ImageFileDetails



def _check_args(path: str, n_copies: int, epsilon_step: float, zipfile: bool, filename: str) -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Provided path: {path} is not directory.")
    if n_copies <= 0:
        raise ValueError("Number of created copies must be positive.")
    if epsilon_step < 0.0 or epsilon_step > 1.0:
        raise ValueError("Epsilon step must be in range <0.0 - 1.0>.")
    if zipfile:
        if not filename:
            raise Exception("In case of exporting dataset to zipfile, filename must be provided.")
        if filename[-4:] != ".zip":
            raise Exception("Provide filename with .zip extension, e.g. \"dataset.zip\"")



def save2zip(img: np.array, img_filename: str, filename: str, path:str) -> None:
    """Save image to the .zip file

    :param img: 2D array which represents an image
    :type img: np.array
    :param img_filename: Name of the image file
    :type img_filename: str
    :param filename: Name of the .zip file
    :type filename: str
    :param path: Path to the directory where .zip file is stored
    :type path: str
    """
    zip_path = filename
    if path:
        zip_path = path+filename

    _, encoded_image = cv2.imencode(".png", img)
    with ZipFile(zip_path, "a") as zip:
        zip.writestr(img_filename, encoded_image.tobytes())
        
def save2directory(img: np.array, img_filename: str, path: str) -> None:
    """Save image to the directory

    :param img: 2D array which represents an image
    :type img: np.array
    :param img_filename: Name of the image file
    :type img_filename: str
    :param path: Path to the output directory
    :type path: str
    """
    img = Image.fromarray(img)
    img.save(path+img_filename)
        
        

def parameters2csv(parameters: List[Dict], path: str, parameters_filename: str) -> None:
    """Save parameters to .csv file

    :param parameters: Parameters for each image
    :type parameters: List[Dict]
    :param path: Path to the output directory
    :type path: str
    :param parameters_filename: Name of the parameters file
    :type parameters_filename: str
    """
    df = pd.DataFrame.from_dict(parameters)
    df.to_csv(path+parameters_filename, encoding='utf-8', index=False)


def generate_balanced_dataset(path: str,
                              n_copies: int,
                              epsilon_range: Tuple[float, float]=(0.0, 1.0), 
                              epsilon_step: float=0.001,
                              size: Tuple[int, int]=(640, 480),
                              brightness: Tuple[int, int]=(80,210),
                              zipfile: bool=False,
                              filename: str=None,
                              save_parameters: bool=True,
                              parameters_filename: str="parameters.csv"
                              ) -> None:
    """Generate balanced dataset and save to the output directory or .zip file.

    :param path: Path where output images or compressed .zip file should be stored
    :type path: str
    :param n_copies: Number of images that has to be created with the same epsilon value.
    :type n_copies: int
    :param epsilon_range: Range of epsilons values used to generate images, defaults to (0.0, 1.0)
    :type epsilon_range: Tuple[float, float], optional
    :param epsilon_step: Step by epsilon value increases every iteration, defaults to 0.001
    :type epsilon_step: float, optional
    :param size: Size of generated images (width, height), defaults to (640, 480)
    :type size: Tuple[int, int], optional
    :param brightness: Brightness range of each pixel, defaults to (80,210)
    :type brightness: Tuple[int, int], optional
    :param zipfile: Set to True if output images should be compressed to .zip file, defaults to False
    :type zipfile: bool, optional
    :param filename: Name of output .zip file. Need to be provided if zipfile is True, defaults to None
    :type filename: str, optional
    :param save_parameters: Set to False if additional file with each image parameters should not be stored, defaults to True
    :type save_parameters: bool, optional
    :param parameters_filename: Name of parameters file, defaults to "parameters.csv"
    :type parameters_filename: str, optional
    """
    _check_args(path, n_copies, epsilon_step, zipfile, filename)
    
    min_epsilon, max_epsilon = epsilon_range
    width, height = size
    max_shift_perc = 0.05
    
    max_width_center_shift =  width * max_shift_perc
    min_width_center = int(width/2 - max_width_center_shift)
    max_width_center = int(width/2 + max_width_center_shift)
    
    max_height_center_shift = height * max_shift_perc
    min_height_center = int(height/2 - max_height_center_shift)
    max_height_center = int(height/2 + max_height_center_shift)
    
    img_index = 0
    parameters: List[Dict] = []
    epsilons = np.arange(start=min_epsilon, stop=max_epsilon, step=epsilon_step)
    for _epsilon in tqdm(epsilons):
        _epsilon = float("{:.3f}".format(_epsilon))
        for _ in range(n_copies):
            ring_center = (random.randint(min_width_center, max_width_center),
                           random.randint(min_height_center, max_height_center))
            img = generate_image(_epsilon, size, ring_center, brightness)
            img_filename = f"{str(img_index).zfill(5)}.png"
            if zipfile:
                save2zip(img, img_filename, filename, path)
            else:
                save2directory(img, img_filename ,path)
            if save_parameters:
                img_details = ImageFileDetails(filename=img_filename,
                                                width=width,
                                                height=height,
                                                epsilon=_epsilon,
                                                ring_center_width=ring_center[0],
                                                ring_center_height=ring_center[1],
                                                min_brightness=brightness[0],
                                                max_brightness=brightness[1])
                parameters.append(img_details.dict())
            img_index += 1
    parameters2csv(parameters, path, parameters_filename)


if __name__ == "__main__":
    pass