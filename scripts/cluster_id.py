#!/usr/bin/env python3
"""
Module containing the preprocessing parameter for the OCR json file(s) before clustering. 
It adds an specific identifier to each text outputs coming from the same picture.
"""


#Import Librairies
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

#Import module from this package
import label_processing.clustering_preprocessing

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'cluster_id.py [-h] -j <json_file> -p <clu_json>'
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
            '-j', '--json_file',
            metavar='',
            type=str,
            required = True,
            help=('Path to the OCR output json file')
            )

    parser.add_argument(
            '-p', '--clu_json',
            metavar='',
            type=str,
            required = True,
            help=('Path to where we want to save the preprocessed json file')
            )

    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsing_args()
    json = args.json_file
    clu_json = args.clu_json
    
    out_dir = os.path.dirname(os.path.realpath(clu_json))
    print(f"\nThe new json_file has been successfully saved in {out_dir}")
    clustering_preprocessing.df_to_json(json, clu_json)


