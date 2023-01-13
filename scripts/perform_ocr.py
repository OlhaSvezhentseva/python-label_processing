#!/usr/bin/env python3
import ocr_pytesseract
import argparse
import os

from pathlib import Path

FILENAME = "ocr_not_preprocessed.txt"
FILENAME_PRE = "ocr_preprocessed.txt"

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'OCR2data.py [-h] -o_OCR </path/to/OCR_output_file/outputOCR.txt> <o_OCR_pre> /path/to/OCR_output_preprocessed_file/outputOCRpre.txt>'
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
            default = os.getcwd(),
            help=('Directory in which the resulting crops and the csv will be stored.'
                  'Default is the user current working directory.')
            )

    
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parsing_args()
    # OCR - without image preprocessing
    crop_dir = args.crop_dir
    if crop_dir[-1] == "/" :
        new_dir = f"{os.path.basename(os.path.dirname(crop_dir))}_ocr"
    else:
        new_dir = f"{os.path.basename(crop_dir)}_ocr"
    path = (f"{crop_dir}/../../{new_dir}/") #parent directory of the cropped pictures
    os.mkdir(path)
    ocr_pytesseract.perform_ocr(crop_dir, path, filename = FILENAME )
    
    # OCR - with image preprocessing
    if not args.no_preprocessing: #gets surpressed when specified in command line
        if crop_dir[-1] == "/" :
            new_dir_pre = f"{os.path.basename(os.path.dirname(crop_dir))}_pre"
        else:
            new_dir_pre = f"{os.path.basename(crop_dir)}_pre"    
        pre_path = (f"{crop_dir}/../../{new_dir}/{new_dir_pre}")
        Path(pre_path).mkdir(parents=True, exist_ok=True)
        ocr_pytesseract.preprocessing(crop_dir, pre_path)
        ocr_pytesseract.perform_ocr(pre_path,new_dir, filename = FILENAME_PRE)