#!/usr/bin/env python3
import ocr_pytesseract
import argparse
import os

from pathlib import Path

FILENAME = "ocr_not_preprocessed.json"
FILENAME_PRE = "ocr_preprocessed.json"

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'perform_ocr.py [-h] [-np] -d <crop-dir>'
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
            '-np', '--no_preprocessing',
            metavar='',
            action=argparse.BooleanOptionalAction,
            help=('optional argument: select whether OCR should also be performed' 
            'with preprocessed pictures ')
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

#NOTE added this so you dont use empty directories
def check_dir(dir) -> None:
    """
    Checks if the directory given as an argument contains jpg files

    Args:
        dir (str): path to directory

    Raises:
        FileNotFoundError: raised if no jpg files are found in directory
    """
    if not any(file_name.endswith('.jpg') for file_name in os.listdir(dir)):
        raise FileNotFoundError(("The directory given does not contain "
                                 "any jpg-files. You might have chosen the wrong "
                                 "directory?")) 
        

if __name__ == "__main__":
    args = parsing_args()
    #Find path to tesseract path
    ocr_pytesseract.find_tesseract()
    # OCR - without image preprocessing
    crop_dir = args.crop_dir
    check_dir(crop_dir)
    if crop_dir[-1] == "/" :
        new_dir = f"{os.path.basename(os.path.dirname(crop_dir))}_ocr"
    else:
        new_dir = f"{os.path.basename(crop_dir)}_ocr"
    path = os.path.join(crop_dir, "../..", new_dir) #parent directory of the cropped pictures
    os.mkdir(path)
    ocr_pytesseract.perform_ocr(crop_dir, path, filename = FILENAME )
    
    # OCR - with image preprocessing
    if not args.no_preprocessing: #gets surpressed when specified in command line
        ocr_pytesseract.perform_ocr(crop_dir, path, filename = FILENAME_PRE,
                                    preprocessing = True)