#!/usr/bin/env python3
"""
Module calculating labels' redundancy of a given text transcription (Ground Truth or OCR generated).
"""

#Import module from this package
import redundancy
#import third party libraries
import argparse
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'redundancy.py [-h] -d <dataset-dir>'
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
            '-d', '--dataset_dir',
            metavar='',
            type=str,
            required = True,
            help=('Path to the dataset containing labels transcriptions')
            )

    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsing_args()
    dataset_dir = args.dataset_dir
    redundancy.per_redundancy(dataset_dir)
