#!/usr/bin/env python3

"""
Module containing the Pytesseract OCR parameters to be performed on the croppend
jpg outputs.
"""

#import third party libraries
import argparse
import os
import glob

from pathlib import Path
from typing import Callable


#Import module from this package
from label_processing.text_recognition import Tesseract, Image, find_tesseract
from label_processing import utils

FILENAME = "ocr_preprocessed.json"
FILENAME_NURI = "ocr_preprocessed_nuri.json"

def parsing_args()-> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'perform_ocr.py [-h] [-v] -d <crop-dir>'
    parser =  argparse.ArgumentParser(description=__doc__,
            add_help = False,
            usage = usage
            )

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-v', '--verbose',
            metavar='',
            action=argparse.BooleanOptionalAction,
            default = False,
            help=('Select whether verbose or quiet mode')
            )
    
    parser.add_argument(
            '-d', '--dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory which contains the cropped jpgs on which the'
                  'ocr is supposed to be applied')
            )

    
    args = parser.parse_args()

    return args
    

def ocr_on_dir(crop_dir: str, new_dir: str,
               verbose_print: Callable) -> list[dict[str,str]]:
    #Initialise Tesseract wrapper
    tesseract = Tesseract()
    
    ocr_results: list = []
    count_qr: int = 0
    total_nuri: int = 0
    for file_path in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        image = Image.read_image(file_path)
        #trying to read the qr_code
        decoded_qr = image.read_qr_code_2()
        if decoded_qr is not None:
            verbose_print(f"Qr-Code detected in {image.filename}\n")
            transcript: dict[str, str] = {"ID": image.filename,
                                          "text": decoded_qr}
            count_qr+=1
        else:
            #Preprocessing
            verbose_print(f"Performing preprocessing on {image.filename}")
            image = image.preprocessing() #preprocessed image
            image.save_image(new_dir)#saving image in new directory
            #OCR
            tesseract.image = image
            verbose_print(f"Performing OCR on {image.filename}\n")
            transcript: dict[str, str] = tesseract.image_to_string()
            #get nuri
            if utils.check_text(transcript["text"]):
                total_nuri+= 1 
            transcript = utils.replace_nuri(transcript)
        ocr_results.append(transcript)
    verbose_print(f"QR-codes read: {count_qr}")
    verbose_print(f"get_nuri: {total_nuri}")
    return ocr_results
        
if __name__ == "__main__":
    args = parsing_args()
    #New function verbose print
    verbose_print: Callable = print if args.verbose else lambda *a, **k: None    
    #Find path to tesseract
    find_tesseract()
    verbose_print("Tesseract succesfully detected.\n")
    crop_dir = args.dir
    utils.check_dir(crop_dir)
    new_dir = utils.generate_filename(crop_dir, "preprocessed")
    #Parent directory of the cropped pictures
    parent_dir = os.path.join(crop_dir, os.pardir) 
    new_dir_path = os.path.join(parent_dir, new_dir)
    Path(new_dir_path).mkdir(parents=True, exist_ok=True)
    
    verbose_print(f"\nPerforming OCR on {os.path.abspath(crop_dir)} .\n")
    result_data = ocr_on_dir(crop_dir,
                             new_dir_path,
                             verbose_print)
    verbose_print((f"\nPreprocessed images have been saved in"
                   f"os.path.abspath{os.path.abspath(new_dir_path)} ."))
    
    verbose_print(f"Saving results in {os.path.abspath(parent_dir)} .")
    utils.save_json(result_data, FILENAME, parent_dir)
        
