"""
Creates two separated json files from the OCR output json file.
One for the NURIs and one of the rest of the transcription.
"""

# Import Librairies
import argparse
import os
import warnings
import json
import ast
import os
#from this package
from label_processing import utils
warnings.filterwarnings('ignore')



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

def json_load(f: str) -> list:
    """
    Loads predictions from the OCR outputs as a json file.

    Args:
        f (str): path to the json file

    Returns:
        python_dict (list): list of dictionaries
    """

    with open(f, 'r') as f:
        data = json.load(f)
        data = json.dumps(data)
        python_dict = ast.literal_eval(data)
    return python_dict


def cut_nuris(json: list) -> list:
    """
    Filters items in the list of dictionaries that are not starting with http.

    Args:
        f (list): list of dictionaries

    Returns:
        a (list): filtered list of dictionaries
    """
    data = json_load(json)
    result = []
    prefix = 'http'
    for item in data:
        if item['text'].startswith(prefix) is False:
            result.append(item)
    a = result
    return a


#does not execute main if the script is imported as a module
if __name__ == '__main__': 
    args = parsing_args()
    json_file = args.json_file
    out_dir = args.saving_directory
    
    #filter and write new json files
    json_data = cut_nuris(json_file)
    new_filename = utils.generate_filename(json_file,"_no_nuris" ,".json")
    utils.save_json(json_data, new_filename, out_dir)

    print(f"The json files have been successfully saved in {out_dir}")

