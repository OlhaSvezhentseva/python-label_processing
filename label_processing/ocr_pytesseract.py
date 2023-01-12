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


# Path to Pytesseract exe file
py.pytesseract.tesseract_cmd = r"/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract"


#---------------------OCR---------------------#


def OCR (new_dir, path, out_dir_OCR = os.getcwd()):
    """
    Perfoms Optical Character Recognition with the pytesseract python librairy on jpg images.
    
    Args:
        new_dir (str): path to the directory where the cropped jpgs' main folder is saved.
        path (str): path to the directory where the cropped jpgs are saved.
        out_dir_OCR (str): path to the target directory to save the OCR outputs.
    """
    print(f"\nPerforming OCR on {os.path.basename(new_dir)}!")
    config=r'--psm 11 --oem 3'
    languages = 'eng+deu+fra+ita+spa+por'
    for images in glob.glob(os.path.join(f"{path}/*.jpg")):
        images_filename = os.path.basename(images)
        print(f"Performing OCR on {os.path.basename(images)}!")
        files = Image.open(images)
        result = py.image_to_string(files, languages, config)
        file1 = open(out_dir_OCR, "a+")
        file1.write(images_filename+"\n")
        file1.write(result+"\n"+"\n")
        file1.close()
    print("\nOCR successful")
    print("DONE!")
