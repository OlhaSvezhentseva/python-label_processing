#!/usr/bin/env python3
"""
Module calculating labels' redundancy
"""
#Import module from this package
import redundancy
#import third party libraries
import argparse
import pandas as pd
import re
import warnings
import os
warnings.filterwarnings('ignore')

FILENAME_TXT = "percentage_red.txt"

def parsing_args() -> argparse.ArgumentParser::
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
            
    parser.add_argument(
            '-o', '--output',
            metavar='',
            type=str,
            required = True,
            help=('Target folder where the text file with the redundancy result is saved')
            )

    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsing_args()
    dataset_dir = args.dataset_dir
    result_dir = args.output
    
    result = redundancy.per_redundancy(dataset_dir)
    out_dir = os.path.realpath(result_dir)

    #Write result in text file
    with open(os.path.join(out_dir,FILENAME_TXT), "w") as text_file:
        text_file.write(('%s%%' % result))
        
    filepath = os.path.join(out_dir, FILENAME_TXT)
    print(f"The txt has been successfully saved in {filepath}")
