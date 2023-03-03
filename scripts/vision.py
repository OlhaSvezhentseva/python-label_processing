#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import text_recognition
import json
import glob
import os

#CREDENTIALS = '/home/leonardo/to_save/Projects/Museum_for_Natural_history/ocr_to_data/total-contact-297417-48ed6585325e.json'
#DIR = '/home/leonardo/to_save/Projects/Museum_for_Natural_history/ocr_to_data/results_ocr/test'
RESULTS_JSON = "ocr_google_vision.json" #TODO make this customizable

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'vision.py [-h] [-np] -d <crop-dir> -c <credentials>'
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
            help=('Path tom the google credentials json file')
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
    text_recognition.check_dir(crop_dir) #check if jpegs exist
    for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        image = text_recognition.VisionApi.read_image(file, credentials)
        ocr_result: dict = image.vision_ocr()
        results_json.append(ocr_result)
    
    filepath = RESULTS_JSON
    #select wheteher it should be saved as utf-8 or ascii
    if encoding == 'utf8':
        with open(filepath, "w", encoding = 'utf8') as f:
            print("utf8")
            json.dump(results_json, f, ensure_ascii=False)
    else:
        with open(filepath, "w", encoding = 'ascii') as f:
            json.dump(results_json, f)

if __name__ == '__main__':
    args = parsing_args()
    main(args.dir, args.credentials)
