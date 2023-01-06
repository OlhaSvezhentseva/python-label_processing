'''
Execute the apply_model script.
Takes as inputs: the path to the jpgs (jpg_dir), the path to the model (model), 
                 the classes used for the model (classes) and 
                 the path to the directory in which the resulting crops and the csv will be stored (out_dir).
Outputs: the labels in the pictures are segmented and cropped out of the picture,
         becoming their own file named after their jpg of origin and class.
'''

import apply_model
import argparse
import os
import urllib3
urllib3.disable_warnings()
#import pandas as pd
#from pandas.core.common import SettingWithCopyWarning
#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

CLS_AMOUNT = 3

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'create_crops.py [-h]'
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
            '-c', '--classes',
            metavar='',
            choices = range(1,3),
            type=int,
            default = 1,
            action="store",
            help=("Optional argument: select which classes should be used according to the model selected." 
                 "1 : only box."
                 "2 : classes (location, nuri, uce, taxonomy, antweb, casent_number, dna, other, collection)."
                 "3 : handwritten/typed."
                 "Default is only box"
            )
    
    parser.add_argument(
            '-j', '--jpg_dir',
            metavar='',
            type=str,
            required = True,
            help='Directory where the jpgs are stored.'
            )
    
    parser.add_argument(
            '-m', '--model',
            metavar='',
            type=str,
            required = True,
            help='Path to the model to be used.'
            )
    
    parser.add_argument(
            '-o', '--out_dir',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the resulting crops and the csv will be stored.'
                  'Default is the user current working directory.'
            )
    )
    
    args = parser.parse_args()

    return args


def get_classtype(class_int):
    """
    Returns the chosen classes.

    Args:
        class_int (int): integer for class selection.

    Returns:
        list: list with the selected classes.
    """
    if class_int == 1:
        return ["box"]
    elif class_int == 2:
        return ["location", "nuri", "uce", "taxonomy", "antweb",
                "casent_number", "dna", "other", "collection"]
    else:
        return ["handwritten", "typed"]

# does not execute main if the script is imported as a module
if __name__ == '__main__': 
    args = parsing_args()
    model_path = args.model
    jpeg_dir = args.jpg_dir
    classes = get_classtype(args.classes)
    
    predictions = apply_model.Predict_Labels(model_path, classes, jpeg_dir)
    
    # 1. Call Model
    model = predictions.get_model()
    # 2. Model Predictions
    df = predictions.class_prediction(model)
    # 3. Filter model predictions and save csv
    df = predictions.clean_predictions(df, out_dir = args.out_dir)
    # 4. Cropping
    apply_model.create_crops(jpeg_dir, df, out_dir = args.out_dir)
