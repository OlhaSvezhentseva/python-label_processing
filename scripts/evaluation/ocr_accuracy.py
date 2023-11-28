#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os

# Suppress warning messages during execution
import warnings
warnings.filterwarnings('ignore')

# Import the necessary module from the 'label_evaluation' module package
from label_evaluation import evaluate_text


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'ocr_accuracy.py [-h] -g <ground_truth> -p <predicted_ocr> -r <results>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the evaluate_text.py module.",
        add_help = False,
        usage = usage)

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
            help=('Path to the ground truth dataset.')
            )

    parser.add_argument(
            '-p', '--predicted_ocr',
            metavar='',
            type=str,
            required = True,
            help=('Path json file OCR output.')
            )

    parser.add_argument(
            '-r', '--results',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Target folder where the accuracy results are saved.\n'
                  'Default is the user current working directory.')
            )

    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    gt = args.ground_truth
    pred = args.predicted_ocr
    folder = args.results


    out_dir = os.path.realpath(folder)
    print(f"\nThe OCR accuracy results have been successfully saved in {out_dir}")
    evaluate_text.evaluate_text_predictions(gt, pred, folder)


