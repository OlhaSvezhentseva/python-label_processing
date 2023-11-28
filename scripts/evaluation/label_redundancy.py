#!/usr/bin/env python3

# Import the necessary module from the 'label_evaluation' module package
from label_evaluation import redundancy

# Import third-party libraries
import argparse
import json
import warnings
import os

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

FILENAME_TXT = "percentage_red.txt"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'label_redundancy.py [-h] -d <dataset-dir> -o <output>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the redundancy.py module.",
        add_help = False,
        usage = usage)

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
            help=('Path to the dataset containing labels transcriptions.')
            )
            
    parser.add_argument(
            '-o', '--output',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Target folder where the text file with the redundancy result is saved\n'
                  'Default is the user current working directory.')
            )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset_dir = args.dataset_dir
    result_dir = args.output
    
    with open(dataset_dir, 'r') as file:
        json_data = json.load(file)

    result = redundancy.per_redundancy(json_data)
    out_dir = os.path.realpath(result_dir)

    #Write result in text file
    with open(os.path.join(out_dir,FILENAME_TXT), "w") as text_file:
        text_file.write(('%s%%' % result))
        
    filepath = os.path.join(out_dir, FILENAME_TXT)
    print(f"The txt has been successfully saved in {filepath}")