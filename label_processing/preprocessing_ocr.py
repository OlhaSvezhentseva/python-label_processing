"""
Module containing the Pytesseract OCR parameters with image preprocessing
to be performed on the _cropped jpg outputs from the segmentation_cropping.py module.
"""

# Import Librairies
from PIL import Image
import pytesseract as py
import os
import cv2
import glob

# Path to Pytesseract exe file
py.pytesseract.tesseract_cmd = r"/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract"


#---------------------Preprocessing---------------------#


def Preprocessing(path, pre_path):
    """
    Preprocesses the cropped images to standardize their quality before applying the OCR on them.
    Saves the preprocessed image into a new _pre folder in the main images directory.
    
    Args:
        path (str): path to the directory where the cropped images are saved.
        pre_path (str): path to the target directory to save the preprocessed images.
    """
    print("\nStart Image Preprocessing")
    for filepath in glob.glob(os.path.join(f"{path}/*.jpg")):
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
        filepath = pre_path + filename
        cv2.imwrite(filepath, pre)
    print(f"\nThe images have been successfully saved in {pre_path}")
    print("Preprocessing successful!")


#---------------------OCR---------------------#

def OCR(pre_path, out_dir_OCR_pre = os.getcwd()):
    """
    Perfoms Optical Character Recognition with the pytesseract python librairy on
    preprocessed images.
    
    Args:
        pre_path (str): path to the directory where the cropped preproceesed images are saved.
        out_dir_OCR_pre (str): path to the target directory to save the OCR outputs.
    """
    config=r'--psm 11 --oem 3'
    languages = 'eng+deu+fra+ita+spa+por'
    for images in glob.glob(os.path.join(f"{pre_path}/*.jpg")):
        images_filename = os.path.basename(images)
        print(f"\nPerforming OCR on {os.path.basename(images)}!")
        files = Image.open(images)
        result = py.image_to_string(files, languages, config)
        file1 = open(out_dir_OCR_pre, "a+")
        file1.write(images_filename+"\n")
        file1.write(result+"\n"+"\n")
        file1.close()
    print("\nOCR successful")
    print("DONE!")
