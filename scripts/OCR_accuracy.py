"""
Module containing the accuracy evaluation parameters of the OCR outputs.
"""


#Import Librairies
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

#Import module from this package
import evaluate_text

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'OCR_accuracy.py [-h] -gt <ground_truth_dataset> -pred <cpredicted_ocr> -pred <cpredicted_ocr> -r <results>'
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
            '-gt', '--ground_truth_dataset',
            metavar='',
            type=str,
            required = True,
            help=('Path to the ground truth dataset')
            )

    parser.add_argument(
            '-pred', '--predicted_ocr',
            metavar='',
            type=str,
            required = True,
            help=('Path json file OCR output')
            )

    parser.add_argument(
            '-r', '--results',
            metavar='',
            type=str,
            required = True,
            help=('Target folder where the accuracy results are saved')
            )

    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsing_args()
    gt = args.ground_truth_dataset
    pred = args.predicted_ocr
    folder = args.results


    out_dir = os.path.dirname(os.path.realpath(folder))
    print(f"\nThe new json_file has been successfully saved in {out_dir}")
    evaluate_text.evaluate_text_predictions(gt, pred, folder)


