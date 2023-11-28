#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import third-party libraries
from __future__ import annotations
import argparse
import glob
import os
import concurrent.futures
from typing import Iterator
import warnings

# Import the necessary module from the 'label_processing' module package
from label_processing import vision, utils

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

#CREDENTIALS = '/home/leonardo/to_save/Projects/Museum_for_Natural_history/ocr_to_data/total-contact-297417-48ed6585325e.json'
#DIR = '/home/leonardo/to_save/Projects/Museum_for_Natural_history/ocr_to_data/results_ocr/test'
RESULTS_JSON = "ocr_google_vision.json"
RESULTS_JSON_BOUNDING = "ocr_google_vision_wbounding.json"
BACKUP_TSV = "ocr_google_vision_backup.tsv"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'vision_api.py [-h] [-np] -d <crop-dir> -c <credentials>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the vision.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Description of the command-line arguments.'
            )
    
    parser.add_argument(
            '-c', '--credentials',
            metavar='',
            type=str,
            required = True,
            help=('Path to the google credentials json file.')
            )
    
    parser.add_argument(
            '-d', '--dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory which contains the cropped jpgs on which the'
                  'ocr is supposed to be applied')
            )

    return parser.parse_args()


def vision_caller(filename: str, credentials: str, backup_file: str) -> dict[str, str]:
    """
    Perform OCR using Google Cloud Vision API on an image file.

    Args:
        filename (str): The path to the input image file.
        credentials (str): The path to the Google Cloud Vision API credentials.
        backup_file (str): The path to save the backup TSV file.

    Returns:
        dict[str, str]: A dictionary containing the OCR results.
    """
    vision_image = vision.VisionApi.read_image(filename, credentials)
    ocr_result: dict = vision_image.vision_ocr()
    with open(backup_file, "w", encoding="utf8") as bf:
        bf.write(f"{ocr_result['ID']}\t{ocr_result['text']}")
    return ocr_result 


def main(crop_dir: str, credentials: str, encoding: str = 'utf8') -> None:
    """
    Perform OCR on a directory containing JPEG images using Google Cloud Vision API.

    Args:
        crop_dir (str): The path to the directory containing JPEG images.
        credentials (str): The path to the Google Cloud Vision API credentials.
        encoding (str, optional): The encoding for saving files. Defaults to 'utf8'.
    """
    results_json = []
    # Check if JPEGs exist in the specified directory
    utils.check_dir(crop_dir)
    
    # Get the list of JPEG filenames
    filenames = [file for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg"))]

    # Run API calls on multiple threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results: Iterator[dict[str, str]] = executor.map(vision_caller,
                                                         filenames,
                                                         [credentials]* len(filenames),
                                                         [BACKUP_TSV]*len(filenames))
    
    results_json = list(results)
    
    # Get the parent_directory
    parent_dir = os.path.join(crop_dir, os.pardir)

    # Select wheteher it should be saved as utf-8 or ascii
    utils.save_json(results_json, RESULTS_JSON_BOUNDING, parent_dir)

    # Without bounding boxes
    json_no_bounding = []
    for entry in results_json:
        entry.pop("bounding_boxes")
        json_no_bounding.append(entry)
    utils.save_json(json_no_bounding, RESULTS_JSON, parent_dir)


if __name__ == '__main__':
    args = parse_arguments()
    exit(main(args.dir, args.credentials))
