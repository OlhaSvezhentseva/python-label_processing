"""
Creates two separated json files from the OCR output json file.
One for the NURIs and one of the rest of the transcription.
"""

# Import Librairies
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
#Import module from this package
from label_postprocessing import nuri_postprocessing


def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'postprocessing_nuri.py [-h] -j <json_file> -d <saving_directory>'
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
            type=str,
            action='store',
            required = True,
            help=('Path to the json file - OCR output.')
            )

    parser.add_argument(
            '-d', '--saving_directory',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the json files will be saved.\n'
                  'Default is the user current working directory.')
            )

    args = parser.parse_args()

    return args



#does not execute main if the script is imported as a module
if __name__ == '__main__': 
    args = parsing_args()
    json = args.json_file
    out_dir = args.saving_directory
    
    #filter and write new json files
    nuri_postprocessing.write_json_with(json, filepath = out_dir)
    nuri_postprocessing.write_json_without(json, filepath = out_dir)

    print(f"The json files have been successfully saved in {out_dir}")

