
#!/usr/bin/env python3
"""
Module containing the accuracy evaluation parameters of the segmentation model.
"""


#Import Librairies
import argparse
import os
import warnings
import pandas as pd
import plotly.io as pio
warnings.filterwarnings('ignore')

#Import module from this package
from label_processing import iou_scores

#Setting filenames as Constants
#TODO change this
FILENAME_CSV = "iou_scores.csv"
FILENAME_BOXPLOT = "iou_box.jpg"
FILENAME_BARCHART = "class_pred.jpg"

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'segmentation_accuracy.py [-h] -g <ground_truth_coord> -p <predicted_coord> -r <results>'
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
            '-g', '--ground_truth_coord',
            metavar='',
            type=str,
            required = True,
            help=('Path to the ground truth coordinates csv')
            )

    parser.add_argument(
            '-p', '--predicted_coord',
            metavar='',
            type=str,
            required = True,
            help=('Path to the predicted coordinates csv')
            )

    parser.add_argument(
            '-r', '--results',
            metavar='',
            type=str,
            required = True,
            help=('Target folder where the iou accuracy results and plots are saved')
            )

    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsing_args()
    gt = args.ground_truth_coord
    pred = args.predicted_coord
    result_dir = args.results


    out_dir = os.path.realpath(result_dir)
    df_gt = pd.read_csv(args.ground_truth_coord)
    df_pred = pd.read_csv(args.predicted_coord)
    
    df_concat = iou_scores.concat_frames(df_gt, df_pred)
    filepath = os.path.join(result_dir, FILENAME_CSV)
    df_concat.to_csv(FILENAME_CSV)
    print(f"The csv has been successfully saved in {filepath}")
    #create boxplot
    fig = iou_scores.box_plot_iou(df_concat)
    boxplot_path = os.path.join(result_dir, FILENAME_BOXPLOT)
    pio.write_image(fig, boxplot_path, format = "jpg")
    print(f"The boxplot has been successfully saved in {boxplot_path}")
    #create barchart
    fig = iou_scores.class_pred(df_concat)
    barchart_path = os.path.join(result_dir, FILENAME_BARCHART)
    pio.write_image(fig, barchart_path, format = "jpg")
    print(f"The boxplot has been successfully saved in {barchart_path}")


