"""
Module containing the Pytesseract OCR parameters to be performed on the _cropped jpg outputs.
"""

#!/usr/bin/env python3
import text_recognition
import utils
import argparse
import os
import json

from pathlib import Path

FILENAME = "ocr_preprocessed.json"
FILENAME_NURI = "ocr_preprocessed_nuri.json"

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'perform_ocr.py [-h] -d <crop-dir>'
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

if __name__ == "__main__":
    args = parsing_args()
    text_recognition.VERBOSE = args.verbose
    #Find path to tesseract
    text_recognition.find_tesseract()
    # OCR - without image preprocessingfrom pathlib import Path
    crop_dir = args.crop_dir
    utils.check_dir(crop_dir)
    new_dir = utils.generate_filename(crop_dir, "ocr")
    #parent directory of the cropped pictures
    parent_dir = os.path.join(crop_dir, os.pardir) 
    new_dir_path = os.path.join(parent_dir, new_dir)
    Path(new_dir_path).mkdir(parents=True, exist_ok=True)
    result_data = text_recognition.perform_tesseract_ocr(crop_dir,
                                                         parent_dir,
                                                         filename = FILENAME,
                                                         dir_name = new_dir_path)
    utils.save_json(result_data, FILENAME, parent_dir)
    #Get the json with regex nuri
    result_data = utils.get_nuri(result_data)
    utils.save_json(result_data, FILENAME_NURI, parent_dir)
        