"""
Module containing the Pytesseract OCR parameters and image preprocessing to be performed on the _cropped jpg outputs from
the segmentation_cropping.py module.
"""

# Import Librairies
from __future__ import annotations
import os
import glob
import cv2
import shutil
import math
import pytesseract as py
import numpy as np
from pathlib import Path
from typing import Callable, Union, Tuple, Optional
from deskew import determine_skew

#Configuarations
CONFIG = r'--psm 6 --oem 3' #configuration for ocr
LANGUAGES = 'eng+deu+fra+ita+spa+por' #specifying languages used for ocr
VERBOSE = False

#new function verbose print
verbose_print: Callable = print if VERBOSE else lambda *a, **k: None

# Path to Pytesseract exe file
def find_tesseract() -> None:
    """
    Searches for the tesseract executable and raises an error if it is not found.
    """
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        raise FileNotFoundError(("Could not find tesseract on your machine!"
                                 "Please read the README for instructions!"))
    else:
        py.pytesseract.tesseract_cmd = tesseract_path


#TODO maybe iput it as a dictionary with keywargs all the important parameters
#also have the image as a return 
class Image():
    """
    A class for image preprocessing.
    """
    def __init__(self, image, path, languages = LANGUAGES, config = CONFIG):
        self.image = image
        self.path = path
        self.filename = os.path.basename(self.path)
        self.languages = languages
        self.config = config
        #preprocessing parameters
    
    @staticmethod
    def read_image(path: str) -> Image:
        """
        Returns instance of preprocessing of a picture.

        Args:
            path (str): path to a jpg file

        Returns:
            Preprocessing: instance of preprocessing
        """
        return Image(cv2.imread(path), path)
        
    #gray scale
    def get_grayscale(self) -> Image:
        image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        return Image(image, self.path,
                     languages = self.languages, config = self.config)

    #blur
    def blur(self) -> Image:
        image = cv2.GaussianBlur(self.image, (5,5), 0)
        return Image(image, self.path,
                     languages = self.languages, config = self.config)

    #noise removal
    def remove_noise(self) -> Image:
        image = cv2.medianBlur(self.image,5)
        return Image(image, self.path,
                     languages = self.languages, config = self.config)
    
    #thresholdingpython json tool utf 8
    def thresholding(self) -> Image:
        image = cv2.threshold(self.image, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return Image(image, self.path,
                     languages = self.languages, config = self.config)
    
    #dilation
    def dilate(self) -> Image:
        kernel = np.ones((5,5),np.uint8)
        image =  cv2.dilate(self.image, kernel, iterations = 1)
        return Image(image, self.path,
                     languages = self.languages, config = self.config)
    #erosion
    def erode(self) -> Image:
        kernel = np.ones((5,5),np.uint8)
        image = cv2.erode(self.image, kernel, iterations = 1)
        return Image(image, self.path,
                     languages = self.languages, config = self.config)
    
    @staticmethod
    def _rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
        ) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = (abs(np.sin(angle_radian) * old_height) 
            + abs(np.cos(angle_radian) * old_width))
        height = (abs(np.sin(angle_radian) * old_width) 
            + abs(np.cos(angle_radian) * old_height))

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)),
                                               int(round(width))),
                              borderValue=background)

    def get_skew_angle(self) -> Optional[np.float64]: #returns either float or None 
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        verbose_print(f"Calculating skew angle for {self.filename}")
        angle = determine_skew(grayscale)
        return angle
        
    def deskew(self, angle: Optional[np.float64]) -> Image:
        verbose_print(f"Rotating {self.filename}")
        rotated = self._rotate(self.image, angle, (0, 0, 0))
        return Image(rotated, self. path,
                     languages = self.languages, config = self.config)


#TODO create a tesseract class
def improved_image_to_string(image: Image) -> Tuple[str, Image]:
    """
    Apply OCR and Image parameters on jpg images.
    
    Args:
        img (str): path to jpgs
        languages (str): OCR available languages
        config (str): any additional custom configuration flags
        that are not available via the pytesseract function.
        
    Returns:
        str: output as string from Tesseract OCR processing.
    """
    #skew angle has to be calculated before preprocessing
    angle = image.get_skew_angle()
    image = image.get_grayscale()
    image = image.thresholding()
    image = image.blur()
    image = image.remove_noise()
    image = image.deskew(angle)
    transcript = py.image_to_string(image.image, image.languages, image.config)
    return transcript, image


def process_string(result_raw: str) -> str:
    """
    Processes the ocr_output by replacing \n with spaces and encoding it to
    ascii and decoding it again to utf-8.

    Args:
        result_raw (str): raw string from pytesseract output

    Returns:
        str: processed string
    """
    processed = result_raw.replace('\n', ' ')
    return processed

def perform_tesseract_ocr(crop_dir: str, path: str, filename: str,
                          dir_name: str) -> None:
    """
    Perfoms Optical Character Recognition with the pytesseract python librairy on jpg images.
    
    Args:
        crop_dir (str): path to the directory where the cropped jpgs' main folder is saved.
        path (str): path to the directory where the cropped jpgs are saved.
        filename (str): json file's filename
    """
    verbose_print(f"\nPerforming OCR on {os.path.basename(crop_dir)}!")
    filepath: str = os.path.join(path,filename)
    ocr_results: list = []
    
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        image_filename = os.path.basename(file)
        verbose_print(f"Performing OCR on {os.path.basename(file)}!")
        image = Image.read_image(file)
        result, image = improved_image_to_string(image)
        #save the preprocessed image
        filename_processed = os.path.join(dir_name, image.filename)
        cv2.imwrite(filename_processed,image.image)
        #remove linebreaks
        result_processed = process_string(result)
        ocr_results.append({"ID": image_filename, "text": result_processed})

    verbose_print("\nOCR successful")
    verbose_print(f"Saving ocr results in {filepath}")
    return ocr_results



    
    

