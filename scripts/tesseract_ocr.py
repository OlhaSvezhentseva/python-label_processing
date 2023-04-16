#!/usr/bin/env python3

"""
Module containing the Pytesseract OCR parameters to be performed on the _cropped jpg outputs.
"""

import utils
import argparse
import os
import json
import glob

from text_recognition import Tesseract, Image, find_tesseract
from pathlib import Path
from typing import Callable

FILENAME = "ocr_preprocessed.json"
FILENAME_NURI = "ocr_preprocessed_nuri.json"

def parsing_args():
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
            '-d', '--crop_dir',
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
    #initialise tesseract wrapper
    tesseract = Tesseract()
    
    ocr_results: list = []
    for file_path in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        #preprocessing
        image = Image.read_image(file_path)
        verbose_print(f"Performing preprocessing on {image.filename}")
        image = image.preprocessing() #preprocessed image
        image.save_image(new_dir)#saving image in new directory
        #ocr
        tesseract.image = image
        verbose_print(f"Performing OCR on {image.filename}")
        transcript: dict[str, str] = tesseract.image_to_string()
        ocr_results.append(transcript)
    
    return ocr_results
        
if __name__ == "__main__":
    args = parsing_args()
    #new function verbose print
    verbose_print: Callable = print if args.verbose else lambda *a, **k: None    
    #Find path to tesseract
    find_tesseract()
    verbose_print("Tesseract succesfully detected.\n")
    
    crop_dir = args.crop_dir
    utils.check_dir(crop_dir)
    new_dir = utils.generate_filename(crop_dir, "preprocessed")
    #parent directory of the cropped pictures
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
    #Get the json with regex nuri
    result_data = utils.get_nuri(result_data)
    utils.save_json(result_data, FILENAME_NURI, parent_dir)
    verbose_print(f"Saving results in {os.path.abspath(parent_dir)} .")
    utils.save_json(result_data, FILENAME, parent_dir)
    #Get the json with regex nuri
    result_data = utils.get_nuri(result_data)
    utils.save_json(result_data, FILENAME_NURI, parent_dir)
        
