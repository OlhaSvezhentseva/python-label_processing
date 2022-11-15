#!/usr/bin/env python3

'''
Takes the csv with the labelnames and the directory with the images as an input 
and creates a new directory with in the current working directory. This directory
contains one subdirectory for each picture in the input directory with the 
cropped labels of every picture as seperate images in each of these directories.
'''

#TODO make this different, so that only one picture is loaded in at a time

import apply_model
import argparse
import os

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
            help='open this help text.'
            )
    parser.add_argument(
            '-c', '--classes',
            metavar='',
            action=argparse.BooleanOptionalAction,
            help='optional argument: select whether classes should be used'
            )
    
    parser.add_argument(
            '-j', '--jpg_dir',
            metavar='',
            type=str,
            required = True,
            help='directory where the collection event jpgs are stored'
            )
    
    parser.add_argument(
            '-m', '--model',
            metavar='',
            type=str,
            required = True,
            help='path to the model to be used'
            )
    
    args = parser.parse_args()

    return args

def get_classtype(class_bool):
    if class_bool:
        classes = ["location", "nuri", "uce", "taxonomy", "antweb",
                "casent_number", "dna", "other", "collection"]
    else:
        classes = ["box"]
    return classes


# does not execute main if the script is imported as a module
if __name__ == '__main__': 
    #parse arguments
    args = parsing_args()
    model_path = args.model
    jpeg_dir = args.jpg_dir
    classes = get_classtype(args.classes)
    
    predictions = apply_model.Predict_Labels(model_path,
                                                              classes, jpeg_dir)
    
    # 2. Call Model
    model = predictions.get_model()
    # 3. Models Predictions
    df = predictions.class_prediction(model)
    # 4. Filter model predictions and save csv
    df = predictions.clean_predictions(df)
    # 5. cropping
    apply_model.create_crops(jpeg_dir, df)
