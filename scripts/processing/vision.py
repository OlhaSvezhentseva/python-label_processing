#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import third-party libraries
from __future__ import annotations
import argparse
import glob
import os
import warnings
import time
import cv2  # Import OpenCV for QR code detection
from google.cloud import vision
from google.oauth2 import service_account

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
        argparse.Namespace: Parsed command-line arguments, including input directories,
        credentials file, output directory, and verbosity flag.
    """
    usage = 'vision.py [-h] [-np] -d <crop dir> -c <credentials> -o <output dir> -v'

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
        required=True,
        help='Directory where the JSON outputs will be saved.'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )

    return parser.parse_args()


def vision_caller(filename: str, credentials: str, output_dir: str, verbose: bool) -> dict[str, str]:
    """
    Perform OCR on an image file using Google Cloud Vision API.

    Args:
        filename (str): Path to the image file.
        credentials (str): Path to the Google Cloud Vision API credentials JSON file.
        output_dir (str): Directory where the backup TSV file will be saved.
        verbose (bool): Flag to enable verbose output.

    Returns:
        dict[str, str]: A dictionary containing the OCR result with 'ID' and 'text'.
    """
    if verbose:
        print(f"[INFO] Processing file: {filename}")

    credentials = service_account.Credentials.from_service_account_file(credentials)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with open(filename, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    if verbose:
        print(f"[INFO] Calling Google Vision API for file: {filename}")

    try:
        response = client.text_detection(image=image)
        texts = response.text_annotations
    except Exception as e:
        print(f"[ERROR] Google Vision API request failed for file {filename}: {e}")
        return {"ID": os.path.basename(filename), "text": "", "error": str(e)}

    ocr_result = {"ID": os.path.basename(filename), "text": texts[0].description if texts else ""}
    backup_file = os.path.join(output_dir, BACKUP_TSV)

    with open(backup_file, "a", encoding="utf8") as bf:
        bf.write(f"{ocr_result['ID']}\t{ocr_result['text']}\n")

    if verbose:
        print(f"[INFO] Finished processing file: {filename}")

    return ocr_result


def detect_qr_code(image_path: str, verbose: bool) -> bool:
    """
    Detect if an image contains a QR code.

    Args:
        image_path (str): Path to the image file.
        verbose (bool): Flag to enable verbose output.

    Returns:
        bool: True if a QR code is detected, False otherwise.
    """
    if not os.path.isfile(image_path):
        if verbose:
            print(f"[ERROR] File not found: {image_path}")
        return False

    image = cv2.imread(image_path)
    if image is None:
        if verbose:
            print(f"[ERROR] Error reading image: {image_path}")
        return False

    qr_detector = cv2.QRCodeDetector()
    try:
        data, bbox, _ = qr_detector.detectAndDecode(image)
        if data:
            if verbose:
                print(f"[INFO] QR code detected in {image_path}")
            return True
    except cv2.error as e:
        if verbose:
            print(f"[ERROR] Error detecting QR code in {image_path}: {e}")

    return False


def main(crop_dir: str, credentials: str, output_dir: str, encoding: str = 'utf8', verbose: bool = False) -> None:
    """
    Perform OCR on all JPEG images in a directory using Google Cloud Vision API.

    Args:
        crop_dir (str): Directory containing the JPEG images to process.
        credentials (str): Path to the Google Cloud Vision API credentials JSON file.
        output_dir (str): Directory where the JSON outputs will be saved.
        encoding (str, optional): Encoding to use for saving files. Defaults to 'utf8'.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Returns:
        None
    """
    start_time = time.time()
    print("Starting OCR process...")
    results_json = []
    utils.check_dir(crop_dir)
    
    # Get the list of JPEG filenames
    filenames = [file for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg"))]
    if verbose:
        print(f"[INFO] Total number of files found: {len(filenames)}")

    filenames = [file for file in filenames if not detect_qr_code(file, verbose)]
    if verbose:
        print(f"[INFO] Number of files to process after filtering QR codes: {len(filenames)}")

    for filename in filenames:
        result = vision_caller(filename, credentials, output_dir, verbose)
        results_json.append(result)

    print("[INFO] OCR process completed.")
    print("[INFO] Saving OCR results...")
    utils.save_json(results_json, RESULTS_JSON_BOUNDING, output_dir)

    json_no_bounding = []
    for entry in results_json:
        entry.pop("bounding_boxes", None)
        json_no_bounding.append(entry)
    utils.save_json(json_no_bounding, RESULTS_JSON, output_dir)

    end_time = time.time()
    duration = end_time - start_time
    print(f"[INFO] Total time taken: {duration} seconds")


if __name__ == '__main__':
    args = parse_arguments()
    vision_caller.processed_count = 0
    exit(main(args.dir, args.credentials, args.output_dir, verbose=args.verbose))
