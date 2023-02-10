"""
Module containing the Pytesseract OCR parameters to be performed on the _cropped jpg outputs from
the segmentation_cropping.py module.
"""

# Import Librairies
from PIL import Image
import pytesseract as py
import os
import glob
import cv2
import json

#Configuarations
CONFIG = r'--psm 11 --oem 3' #configuration for ocr
LANGUAGES = 'eng+deu+fra+ita+spa+por' #specifying languages used for ocr

# Path to Pytesseract exe file
#py.pytesseract.tesseract_cmd = r"/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract"



def preprocessing(crop_dir: str, pre_path: str) -> None:
    """
    Preprocesses the cropped images to standardize their quality before applying the OCR on them.
    Saves the preprocessed image into a new _pre folder in the main images directory.
    
    Args:
        path (str): path to the directory where the cropped images are saved.
        pre_path (str): path to the target directory to save the preprocessed images.
    """
    print("\nStart Image Preprocessing!\n")
    for filepath in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        print(f"Performing preprocessing on {os.path.basename(filepath)}!")
        filename = os.path.basename(filepath)
        img = cv2.imread(filepath)
        # grayscale region within bounding box
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # resize image to three times as large as original for better readability
        gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        # perform gaussian blur to smoothen image
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        # threshold the image using Otsus method to preprocess for pytesseract
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        #perfrom bitwise not to flip image to black text on white background
        pre = cv2.bitwise_not(thresh)
        #saving the preprocessing images
        filepath = f"{pre_path}/{filename}"
        cv2.imwrite(filepath, pre)
    print(f"\nThe images have been successfully saved in {pre_path}")
    print("Preprocessing successful!\n")


#---------------------OCR---------------------#

def perform_ocr(crop_dir: str, path: str, filename:str) -> None:
    """
    Perfoms Optical Character Recognition with the pytesseract python librairy on jpg images.
    
    Args:
        crop_dir (str): path to the directory where the cropped jpgs' main folder is saved.
        path (str): path to the directory where the cropped jpgs are saved.
        out_dir_OCR (str): path to the target directory to save the OCR outputs.
    """
    print(f"\nPerforming OCR on {os.path.basename(crop_dir)}!")
    filepath: str = f"{path}/{filename}"
    
    
    ocr_results: list = []
    
    for image in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        image_filename = os.path.basename(image)
        print(f"Performing OCR on {os.path.basename(image)}!")
        files = Image.open(image)
        result = py.image_to_string(files, LANGUAGES, CONFIG)
        #remove linebreaks
        result = result.replace('\n', '')
        ocr_results.append({"ID": image_filename, "text": result})

    print("\nOCR successful")
    
    with open(filepath, "w") as f:
        json.dump(ocr_results, f)

    print("DONE!")

