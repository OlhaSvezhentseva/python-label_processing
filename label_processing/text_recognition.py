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
import io
import pytesseract as py
import numpy as np
from google.cloud import vision

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


#TODO maybe iput it as a dictionary with keywargs all the important parameters
#also have the image as a return 
class Preprocessing():
    """
    A class for preprossecing an image
    """
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
            config (str): any additional custom configuration flags
            that are not available via the pytesseract function.
            
        Returns:
            str: output as string from Tesseract OCR processing.
        """
        self.get_grayscale()
        self.thresholding()
        #self.opening()
        #self.canny()
        return py.image_to_string(self.image, self.languages, self.config)


#---------------------Tesseract-OCR---------------------#


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
        exports the credentials json, by adding it as an enviroment variable
        in your shell

        Args:
            credentials (str): path to the credentials json file
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials

    @staticmethod
    def read_image(path: str, credentials: str, encoding: str = 'utf8') -> VisionApi:
        """
        reads an image with io and returns it as an instance of the VisionApi
        class

        Args:
            path (str): path to image
            credentials (str): path to the credentials json file
            encoding (str, optional): choose in which encoding th result will 
            be saved (ascii or utf-8). Defaults to 'utf8'.

        Returns:
            VisionApi: Instance of the VisionApi class
        """
        with io.open(path, 'rb') as image_file:
            image = image_file.read()
        return VisionApi(path, image, credentials, encoding)
    
    def process_string(self, result_raw: str) -> str:
        """
        processes the google vision ocr output and replaces newlines by spaces
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
        performs the actual API call, does error handling and returns the 
        transcription already processed

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
                f'{response.error.message}\nFor more info on error messages, '
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
