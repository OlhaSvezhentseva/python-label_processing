#!/usr/bin/env python 
"""
Tries to recognize the background color of a picture and checks if it exceeds
a given threshold. If it exceeds the threshold it moves the corresponding pictures
into a newly created directory
"""

#Import Librairies
import sys
import os
import argparse
import glob
from pathlib import Path
#Import module from this package
from label_processing.backgroundcolor_detection import BackgroundColorDetector


def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'background_color.py [-h] -d <crop-dir>'
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
            '-d', '--dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory which contains the cropped jpgs on which the'
                  'ocr is supposed to be applied.')
            )
    args = parser.parse_args()

    return args

def main(dir: Path) -> None:
    path_non_otsu = Path('for_adaptive')
    path_non_otsu.mkdir(parents=True, exist_ok=True)
    for image_path in glob.glob(os.path.join(dir, "*.jpg")):
        image_path = Path(image_path)
        detector = BackgroundColorDetector(str(image_path), 150)
        if not detector.decide():
            print(f"moving {image_path.name} to {path_non_otsu}")
            os.replace(image_path, os.path.join(path_non_otsu, image_path.name))
    
if __name__ == "__main__":
    args = parsing_args()
    main(args.dir)
