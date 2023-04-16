#!/usr/bin/env python3

'''
Execute the segmentation_cropping.py module.
'''

#Import module from this package
import segmentation_cropping
#import third party libraries
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'crop_seg.py [-h] [-c N] -m <model/number> -j </path/to/jpgs> -o </path/to/jpgs_outputs> '
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
            help=('Directory in which the resulting crops and the csv will be stored.\n'
                  'Default is the user current working directory.')
            )
    
    parser.add_argument(
            '-j', '--jpg_dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the jpgs are stored.')
            )
    
    parser.add_argument( 
            '-m', '--model',
            metavar='',
            choices = range(1,4),
            type=int,
            default = 1,
            action='store',
            help=('Optional argument: select which segmenmtation-model should be used.\n'
                 '1 : only box.\n'
                 '2 : classes (location, nuri, uce, taxonomy, antweb, casent_number, dna, other, collection).\n'
                 '3 : handwritten/typed.\n'
                 'Default is only box')
            )
    
    args = parser.parse_args()

    return args

def get_modelnum(model_int: int) -> str:
    """
    Returns the chosen model.

    Args:
         model_int (int): integer for model selection.

    Returns:
         path (str): path to the selected model.
    """
    script_dir = os.path.dirname(__file__)
    rel_path1 = "../models/model_labels_box.pth"
    model_file1 = os.path.join(script_dir, rel_path1)
    rel_path2 = "../models/model_labels_class.pth"
    model_file2 = os.path.join(script_dir, rel_path2)
    rel_path3 = "../models/model_labels_h_t.pth"
    model_file3 = os.path.join(script_dir, rel_path3)
    if model_int == 1:
        return model_file1
    elif model_int == 2:
        return model_file2
    else:
        return model_file3

def get_classtype(model_int: int) -> list:
    """
    Returns the chosen classes.

    Args:
        class_int (int): integer for class selection.

    Returns:
        list: list with the selected classes.
    """
    if model_int == 1:
        return ["box"]
    elif model_int == 2:
        return ["location", "nuri", "uce", "taxonomy", "antweb",
                "casent_number", "dna", "other", "collection"]
    else:
        return ["handwritten", "typed"]

# does not execute main if the script is imported as a module
if __name__ == '__main__': 
    args = parsing_args()
    model_path = get_modelnum(args.model)
    jpeg_dir = args.jpg_dir
    classes = get_classtype(args.model)
    out_dir = args.out_dir
    
    predictions = segmentation_cropping.Predict_Labels(model_path, classes, jpeg_dir)
    
    # 1. Call Model
    model = predictions.get_model()
    
    # 2. Model Predictions
    df = predictions.class_prediction(model)
    
    # 3. Filter model predictions and save csv
    df = predictions.clean_predictions(df, out_dir = out_dir)
    
    # 4. Cropping
    segmentation_cropping.create_crops(jpeg_dir, df, out_dir = out_dir)
    

