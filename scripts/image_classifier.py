'''
Execute the tensorflow_classifier.py module.
'''

#Import module from this package
from label_processing import tensorflow_classifier
#Import third party libraries
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'image_classifier.py [-h] [-c N] -j </path/to/jpgs> -o </path/to/jpgs_outputs> '
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
            help=('Directory in which the resulting classified pictures and the csv will be stored.\n'
                  'Default is the user current working directory.')
            )
    
    parser.add_argument(
            '-j', '--jpg_dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the jpgs are stored.')
            )
    
    args = parser.parse_args()

    return args


model = "../models/model_classifier"

# does not execute main if the script is imported as a module
if __name__ == '__main__': 
    args = parsing_args()
    model_path = model
    class_names = ['handwritten', 'to_crop', 'typed']
    jpeg_dir = args.jpg_dir
    out_dir = args.out_dir
        
    # 1. Call Model
    model = tensorflow_classifier.get_model(model_path)
    
    # 2. Model Predictions and save csv
    df = tensorflow_classifier.class_prediction(model, class_names, jpeg_dir, out_dir = out_dir)
    
    # 3. Save classified pictures
    tensorflow_classifier.filter_pictures(jpeg_dir, df, out_dir = out_dir)
