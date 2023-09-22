#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Performs the Google Vision OCR on the segmented labels by calling the API and returns it as a json file. 
'''

#Import Librairies
from __future__ import annotations
import argparse
import glob
import os
import concurrent.futures
from typing import Iterator
##Import module from this package
from label_processing import vision, utils

#CREDENTIALS = '/home/leonardo/to_save/Projects/Museum_for_Natural_history/ocr_to_data/total-contact-297417-48ed6585325e.json'
#DIR = '/home/leonardo/to_save/Projects/Museum_for_Natural_history/ocr_to_data/results_ocr/test'
RESULTS_JSON = "ocr_google_vision.json"
RESULTS_JSON_BOUNDING = "ocr_google_vision_wbounding.json"
BACKUP_TSV = "ocr_google_vision_backup.tsv"

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'vision_api.py [-h] [-np] -d <crop-dir> -c <credentials>'
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
            '-c', '--credentials',
            metavar='',
            type=str,
            required = True,
            help=('Path to the google credentials json file')
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

def vision_caller(filename: str, credentials: str, backup_file: str) -> dict[str, str]:
    """
    Perform OCR using Google Cloud Vision API on an image file.

    Args:
        filename (str): The path to the input image file.
        credentials (str): The path to the Google Cloud Vision API credentials.

    Returns:
        dict[str, str]: A dictionary containing the OCR results.
    """
    vision_image = vision.VisionApi.read_image(filename, credentials)
    ocr_result: dict = vision_image.vision_ocr()
    with open(backup_file, "w", encoding="utf8") as bf:
        bf.write(f"{ocr_result['ID']}\t{ocr_result['text']}")
    return ocr_result 


def main(crop_dir: str, credentials: str,
                       encoding: str = 'utf8') -> None:
    """
    Performs the ocr on a dir containing jpgs

    Args:
        crop_dir (str): _description_
        credentials (str): _description_
        encoding (str, optional): _description_. Defaults to 'utf8'.
    """
    
    results_json = []
    utils.check_dir(crop_dir) #Check if jpegs exist
    filenames = [file for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg"))]
    #run api calls on multiple threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results: Iterator[dict[str, str]] = executor.map(vision_caller,
                                                         filenames,
                                                         [credentials]* len(filenames),
                                                         [BACKUP_TSV]*len(filenames))
    
    results_json = list(results)
    
    parent_dir = os.path.join(crop_dir, os.pardir) #Get the parent_directory
    #Select wheteher it should be saved as utf-8 or ascii
    utils.save_json(results_json, RESULTS_JSON_BOUNDING, parent_dir)
    #without bounding boxes
    json_no_bounding = []
    for entry in results_json:
        entry.pop("bounding_boxes")
        json_no_bounding.append(entry)
    utils.save_json(json_no_bounding, RESULTS_JSON, parent_dir)
        
    


if __name__ == '__main__':
    args = parsing_args()
    exit(main(args.dir, args.credentials))
