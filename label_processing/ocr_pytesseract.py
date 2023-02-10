"""
Module containing the Pytesseract OCR parameters to be performed on the _cropped jpg outputs from
the segmentation_cropping.py module.
"""

# Import Librairies
from __future__ import annotations
import os
import glob
import cv2
import json
import shutil
import re
import pytesseract as py
import numpy as np

#Configuarations
CONFIG = r'--psm 11 --oem 3' #configuration for ocr
LANGUAGES = 'eng+deu+fra+ita+spa+por' #specifying languages used for ocr

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


class Preprocessing():
    def __init__(self, image, languages = LANGUAGES, config = CONFIG):
        self.image = image
        self.languages = languages
        self.config = config
    
    @staticmethod
    def read_image(path: str) -> Preprocessing:
        """
        Returns instance of Preprocessing of a picture.

        Args:
            path (str): path to a jpg file

        Returns:
            Preprocessing: instance of Preprossing
        """
        return Preprocessing(cv2.imread(path))
        
    #gray scale
    def get_grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    #noise removal
    def remove_noise(self):
        self.image = cv2.medianBlur(self.image,5)
    
    #thresholding
    def thresholding(self):
        self.image = cv2.threshold(self.image, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #dilation
    def dilate(self):
        kernel = np.ones((5,5),np.uint8)
        self.image =  cv2.dilate(self.image, kernel, iterations = 1)
        
    #erosion
    def erode(self):
        kernel = np.ones((5,5),np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations = 1)

    #opening - erosion followed by dilation
    def opening(self):
        kernel = np.ones((5,5),np.uint8)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)

    #canny edge detection
    def canny(self):
        self.image = cv2.Canny(self.image, 100, 200)

    #skew correction
    def deskew(self):
        coords = np.column_stack(np.where(self.image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        self.image = rotated

    #template matching
    def match_template(self, template):
        self.image =  cv2.matchTemplate(self.image, template,
                                        cv2.TM_CCOEFF_NORMED)

    def improved_image_to_string(self) -> str:
        """
        Apply OCR and preprocessing parameters on jpg images.
        
        Args:
            img (str): path to jpgs
            languages (str): OCR available languages
            config (str): any additional custom configuration flags that are not available via the pytesseract function.
            
        Returns:
            str: output as string from Tesseract OCR processing.
        """
        self.get_grayscale()
        self.thresholding()
        #self.opening()
        #self.canny()
        return py.image_to_string(self.image, self.languages, self.config)


#---------------------OCR---------------------#


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
    processed = processed.encode("ascii", "ignore")
    processed = processed.decode()
    processed = re.sub('\s\s+', ' ', processed)
    return processed

def perform_ocr(crop_dir: str, path: str, filename: str,
                preprocessing: bool = True) -> None:
    """
    Perfoms Optical Character Recognition with the pytesseract python librairy on jpg images.
    
    Args:
        crop_dir (str): path to the directory where the cropped jpgs' main folder is saved.
        path (str): path to the directory where the cropped jpgs are saved.
        filename (str): json file's filename
    """
    print(f"\nPerforming OCR on {os.path.basename(crop_dir)}!")
    filepath: str = f"{path}/{filename}"
    
    ocr_results: list = []
    
    for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        image_filename = os.path.basename(file)
        print(f"Performing OCR on {os.path.basename(file)}!")
        if preprocessing:
            image = Preprocessing.read_image(file)
            result = image.improved_image_to_string()
        else:
            image = cv2.imread(file)
            result = py.image_to_string(image, LANGUAGES, CONFIG)
        #remove linebreaks
        result_processed = process_string(result)
        ocr_results.append({"ID": image_filename, "text": result_processed})

    print("\nOCR successful")
    
    with open(filepath, "w") as f:
        json.dump(ocr_results, f)

    print("DONE!")
