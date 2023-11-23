#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os
import time
import warnings
import glob
import pandas as pd
from pathlib import Path

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

# Import the necessary module from the 'label_processing' module package
import label_processing.segmentation_cropping as scrop
import label_processing.utils
from label_processing.segmentation_cropping import create_crops

THRESHOLD = 0.8
PROCESSES = 12

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'crop_seg.py [-h] [-c N] [-np N] -j </path/to/jpgs> -o </path/to/jpgs_outputs>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the segmentation_cropping.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Description of the command-line arguments.'
            )
    
    parser.add_argument(
            '-o', '--out_dir',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the resulting crops and the csv will be stored.\n'
                  'Default is the user current working directory.')
            )
    
    parser.add_argument(
            '-j', '--jpg_dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the jpgs are stored.')
            )

    return parser.parse_args()


# does not execute main if the script is imported as a module
if __name__ == '__main__': 
    start = time.perf_counter()
    args = parse_arguments()

    # Get model
    script_dir = os.path.dirname(__file__)
    rel_path = "../models/model_segmentation_label.pth"
    model_path = os.path.join(script_dir, rel_path)

    jpg_dir = Path(args.jpg_dir)
    classes = "label"
    out_dir = args.out_dir
    
    predictor = scrop.PredictLabel(model_path, classes)
    
    # 1. Model Predictions
    df = scrop.prediction_parallel(jpg_dir,predictor, PROCESSES)
    finish = time.perf_counter()

    # 2. Filter model predictions and save csv
    df = scrop.clean_predictions(jpg_dir, df, THRESHOLD, out_dir = out_dir)
    print(f"Finished in {round(finish-start, 2)} second(s)")

    # 3. Cropping
    start = time.perf_counter()
    create_crops(jpg_dir, df, out_dir = out_dir)
    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} second(s)")
