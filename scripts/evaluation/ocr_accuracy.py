#!/usr/bin/env python3
"""
Module containing the accuracy evaluation parameters of the OCR outputs.
"""

#Import Librairies
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
#Import module from this package
from label_evaluation import evaluate_text

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'ocr_accuracy.py [-h] -g <ground_truth> -p <predicted_ocr> -r <results>'
    parser =  argparse.ArgumentParser(description=__doc__,
            add_help = False,
            usage = usage
            )

    parser.add_argument(
            '-h', '--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-g', '--ground_truth',
            metavar='',
            type=str,
            required = True,
            help=('Path to the ground truth dataset')
            )

    parser.add_argument(
            '-p', '--predicted_ocr',
            metavar='',
            type=str,
            required = True,
            help=('Path json file OCR output')
            )

    parser.add_argument(
            '-r', '--results',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Target folder where the accuracy results are saved.\n'
                  'Default is the user current working directory.')
            )

    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsing_args()
    gt = args.ground_truth
    pred = args.predicted_ocr
    folder = args.results


    out_dir = os.path.realpath(folder)
    print(f"\nThe OCR accuracy results have been successfully saved in {out_dir}")
    evaluate_text.evaluate_text_predictions(gt, pred, folder)


