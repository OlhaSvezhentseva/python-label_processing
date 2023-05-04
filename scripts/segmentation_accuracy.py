"""
Module containing the accuracy evaluation parameters of the OCR outputs.
"""


#Import Librairies
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

#Import module from this package
import iou_scores

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'segmentation_accuracy.py [-h] -gt <ground_truth_coord> -pred <predicted_coord> -r <results>'
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
            '-gt', '--ground_truth_coord',
            metavar='',
            type=str,
            required = True,
            help=('Path to the ground truth coordinates csv')
            )

    parser.add_argument(
            '-pred', '--predicted_coord',
            metavar='',
            type=str,
            required = True,
            help=('Path to the predicted coordinates csv')
            )

    parser.add_argument(
            '-r', '--results',
            metavar='',
            type=str,
            required = True,
            help=('Target folder where the iou accuracy results and plots are saved')
            )

    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsing_args()
    gt = args.ground_truth_coord
    pred = args.predicted_coord
    folder = args.results


    out_dir = os.path.dirname(os.path.realpath(folder))
    print(f"\nThe new json_file has been successfully saved in {out_dir}")
    iou_scores.accuracy_segmentation(pred, gt, folder)


