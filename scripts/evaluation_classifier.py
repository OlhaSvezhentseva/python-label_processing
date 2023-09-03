'''
Execute the accuracy_classifier.py module.
'''

#Import module from this package
from label_evaluation import accuracy_classifier
#Import third party libraries
import argparse
import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'accuracy_classifier.py [-h] [-c N] -d </path/to/gt_dataframe> -o </path/to/outputs> '
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
            '-o', '--out_dir',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the accuracy scores and plot will be stored.\n'
                'Default is the user current working directory.')
            )
    
    parser.add_argument(
            '-d', '--df',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the csv is stored.')
            )
    
    args = parser.parse_args()

    return args



# does not execute main if the script is imported as a module
if __name__ == '__main__': 
    args = parsing_args()
    target = ['typed', 'to_crop', 'handwritten']
    out_dir = args.out_dir
    df = pd.read_csv(args.df, sep=';')

    pred = df["pred"]
    gt = df["gt"]

    # 1. Accuracy Scores
    accuracy_classifier.metrics(target, pred, gt, out_dir = out_dir)
    
    # 2. Confusion Matrix
    accuracy_classifier.cm(target, pred, gt, out_dir = out_dir)
