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
import time
import threading

# Import the necessary module from the 'label_processing' module package
from label_processing import vision, utils

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

RESULTS_JSON = "ocr_google_vision.json"
RESULTS_JSON_BOUNDING = "ocr_google_vision_wbounding.json"
BACKUP_TSV = "ocr_google_vision_backup.tsv"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'vision_api.py [-h] [-np] -d <crop-dir> -c <credentials> -o <output_dir>'

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
    
    parser.add_argument(
            '-o', '--output_dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the json outputs will be saved.')
            )

    return parser.parse_args()


def vision_caller(filename: str, credentials: str, output_dir: str, lock: threading.Lock) -> dict[str, str]:
    """
    Perform OCR using Google Cloud Vision API on an image file.

    Args:
        filename (str): The path to the input image file.
        credentials (str): The path to the Google Cloud Vision API credentials.
        output_dir (str): The directory where the backup TSV file will be saved.
        lock (threading.Lock): A lock for thread-safe printing.

    Returns:
        dict[str, str]: A dictionary containing the OCR results.
    """
    vision_image = vision.VisionApi.read_image(filename, credentials)
    ocr_result: dict = vision_image.vision_ocr()
    backup_file = os.path.join(output_dir, BACKUP_TSV)
    
    with lock:
        vision_caller.processed_count += 1
        if vision_caller.processed_count % 1000 == 0:
            print(f"Processed {vision_caller.processed_count} images...")
    
    with open(backup_file, "w", encoding="utf8") as bf:
        bf.write(f"{ocr_result['ID']}\t{ocr_result['text']}")
    
    return ocr_result


def main(crop_dir: str, credentials: str, output_dir: str, encoding: str = 'utf8') -> None:
    """
    Perform OCR on a directory containing JPEG images using Google Cloud Vision API.

    Args:
        crop_dir (str): The path to the directory containing JPEG images.
        credentials (str): The path to the Google Cloud Vision API credentials.
        encoding (str, optional): The encoding for saving files. Defaults to 'utf8'.
    """
    start_time = time.time()
    print("Starting OCR process...")
    results_json = []
    # Check if JPEGs exist in the specified directory
    utils.check_dir(crop_dir)
    
    # Get the list of JPEG filenames
    filenames = [file for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg"))]

    # Run API calls on multiple threads
    num_files = len(filenames)
    print(f"Number of files to process: {num_files}")

    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results: Iterator[dict[str, str]] = executor.map(
            vision_caller,
            filenames,
            [credentials]* len(filenames),
            [output_dir]*len(filenames),
            [lock]*len(filenames)
        )
    
    results_json = list(results)

    print("OCR process completed.")

    # Select wheteher it should be saved as utf-8 or ascii
    print("Saving OCR results...")
    utils.save_json(results_json, RESULTS_JSON_BOUNDING, output_dir)

    # Without bounding boxes
    json_no_bounding = []
    for entry in results_json:
        entry.pop("bounding_boxes")
        json_no_bounding.append(entry)
    utils.save_json(json_no_bounding, RESULTS_JSON, output_dir)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time taken: {duration} seconds")


if __name__ == '__main__':
    args = parse_arguments()
    exit(main(args.dir, args.credentials, args.output_dir))
