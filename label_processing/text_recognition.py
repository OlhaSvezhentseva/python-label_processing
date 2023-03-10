"""
Module containing the Pytesseract OCR parameters and image preprocessing to be performed on the _cropped jpg outputs from
the segmentation_cropping.py module.
"""

# Import Librairies
from __future__ import annotations
import os
import glob
import cv2
import json
import shutil
import math
import re
import io
import pytesseract as py
import numpy as np
from pathlib import Path
from typing import Callable, Union, Tuple, Optional
from google.cloud import vision
from deskew import determine_skew

#Configuarations
CONFIG = r'--psm 6 --oem 3' #configuration for ocr
LANGUAGES = 'eng+deu+fra+ita+spa+por' #specifying languages used for ocr
VERBOSE = True

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

#---------------------Tesseract-OCR---------------------#

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
    #processed = processed.encode("ascii", "ignore") 
    #processed = processed.decode()
    #processed = re.sub('\s\s+', ' ', processed)
    return processed

def perform_tesseract_ocr(crop_dir: str, path: str, filename: str,
                preprocessing: bool = True) -> None:
    """
    Perfoms Optical Character Recognition with the pytesseract python librairy on jpg images.
    
    Args:
        crop_dir (str): path to the directory where the cropped jpgs' main folder is saved.
        path (str): path to the directory where the cropped jpgs are saved.
        filename (str): json file's filename
    """
    print(f"\nPerforming OCR on {os.path.basename(crop_dir)}!")
    filepath: str = os.path.join(path,filename)
    ocr_results: list = []

    dir_name = "preprocessing"
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    
    for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        image_filename = os.path.basename(file)
        verbose_print(f"Performing OCR on {os.path.basename(file)}!")
        if preprocessing:
            image = Image.read_image(file)
            result, image = improved_image_to_string(image)
            filename = os.path.join(dir_name, image.filename)
            cv2.imwrite(filename,image.image)
        else:
            image = cv2.imread(file)
            result = py.image_to_string(image, LANGUAGES, CONFIG)
        #remove linebreaks
        result_processed = process_string(result)
        ocr_results.append({"ID": image_filename, "text": result_processed})

    print("\nOCR successful")
    verbose_print(f"Saving ocr results in {filepath}")
    with open(filepath, "w", encoding = 'utf8') as f:
            json.dump(ocr_results, f, ensure_ascii=False)
    return ocr_results

#---------------------Vision-OCR---------------------#

class VisionApi():
    """
    Class concerning the Google Vision API performed on a directory.
    """

    def __init__(self, path: str, image: bytes, credentials: str,
                 encoding: str) -> None:
        VisionApi.export(credentials) #check credententials
        self.image = image
        self.path = path
        self.encoding = encoding

    @staticmethod            
    def export(credentials: str) -> None:
        """
        Exports the credentials json, by adding it as an environment variable
        in your shell.

        Args:
            credentials (str): path to the credentials json file
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials

    @staticmethod
    def read_image(path: str, credentials: str, encoding: str = 'utf8') -> VisionApi:
        """
        Reads an image with io and returns it as an instance of the VisionApi
        class.

        Args:
            path (str): path to image
            credentials (str): path to the credentials json file
            encoding (str, optional): choose in which encoding th result will 
            be saved (ascii or utf-8). defaults to 'utf8'

        Returns:
            VisionApi: Instance of the VisionApi class
        """
        with io.open(path, 'rb') as image_file:
            image = image_file.read()
        return VisionApi(path, image, credentials, encoding)
    
    def process_string(self, result_raw: str) -> str:
        """
        Processes the google vision ocr output and replaces newlines by spaces
        and if specified turns sting from unicode into ascii encoding.

        Args:
            result_raw (str): the raw output string directly from google_vision

        Returns:
            str: processed string
        """
        processed = result_raw.replace('\n', ' ')
        if self.encoding == "ascii":
            #turning it to ascii
            processed = processed.encode("ascii", "ignore")
            return processed.decode()
        else:
            return processed
        
    def vision_ocr(self) -> dict[str, str]:
        """
        Performs the actual API call, does error handling and returns the 
        transcription already processed.

        Raises:
            Exception: raises exception if API does not respond

        Returns:
            dict[str, str]: dictionary with the filename and the transcript
        """
        client = vision.ImageAnnotatorClient()
        vision_image = vision.Image(content=self.image)
        response = client.text_detection(image=vision_image)
        single_transcripts = response.text_annotations #get the ocr results
        #list of transcripts
        transcripts = [str(transcript.description) for transcript in single_transcripts]
        #create string of transcripts
        transcript = self.process_string(transcripts[0])
        #get filename
        filename = os.path.basename(self.path)
        if response.error.message:
            raise Exception(
                f'{response.error.message}\nFor more icheck_IDnfo on error messages, '
                'check:  https://cloud.google.com/apis/design/errors')
        return {'ID' : filename, 'text': transcript}
    
    
def check_dir(dir) -> None:
    """
    Checks if the directory given as an argument contains jpg files.

    Args:
        dir (str): path to directory

    Raises:
        FileNotFoundError: raised if no jpg files are found in directory
    """
    if not any(file_name.endswith('.jpg') for file_name in os.listdir(dir)):
        raise FileNotFoundError(("The directory given does not contain "
                                 "any jpg-files. You might have chosen the wrong "
                                 "directory?")) 
        

def check_text(transcript: str) -> bool:
    pattern = re.compile(r"/u/|http|u/|coll|mfn|/|/u|URI") #search for NURI patterns in "text"
    match = pattern.search(transcript)
    return True if match else False

def get_nuri(data: list[dict[str, str]]) -> list[dict[str, str]]:
    new_data=data.copy()
    reg = re.compile(r"_u_[A-Za-z0-9]+") #search for NURI number in "ID"
    for item, new_item in zip(data, new_data):
        findString = item["text"]
        findNURI = item["ID"]
        if check_text(findString): #checks if label is a NURI - True/False
            try:
                NURI = reg.search(findNURI).group()
                replaceString = "http://coll.mfn-berlin.de/u/"+ NURI[3:]
                new_item["text"] = replaceString #replace "text" with NURI patterns with 
                                                    #formatted "ID"
            except AttributeError:
                    pass
    return new_data
