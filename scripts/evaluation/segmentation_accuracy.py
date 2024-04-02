#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os
import warnings
import pandas as pd
import plotly.io as pio
import time

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

# Import the necessary module from the 'label_evaluation' module package
from label_evaluation import iou_scores


#Setting filenames as Constants
FILENAME_CSV = "iou_scores.csv"
FILENAME_BOXPLOT = "iou_box.jpg"
FILENAME_BARCHART = "class_pred.jpg"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'segmentation_accuracy.py [-h] -g <ground_truth_coord> -p <predicted_coord> -r <results>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the iou_scores.py module.",
        add_help = False,
        usage = usage)

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
            help=('Path to the ground truth coordinates csv.')
            )

    parser.add_argument(
            '-p', '--predicted_coord',
            metavar='',
            type=str,
            required = True,
            help=('Path to the predicted coordinates csv.')
            )

    parser.add_argument(
            '-r', '--results',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Target folder where the iou accuracy results and plots are saved.\n'
                  'Default is the user current working directory.')
            )

    
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()
    gt = args.ground_truth_coord
    pred = args.predicted_coord
    result_dir = args.results
    out_dir = os.path.realpath(result_dir)
    df_gt = pd.read_csv(args.ground_truth_coord)
    df_pred = pd.read_csv(args.predicted_coord)
    
    # create csv
    df_concat = iou_scores.concat_frames(df_gt, df_pred)
    csv_filepath = os.path.join(result_dir, FILENAME_CSV)
    df_concat.to_csv(csv_filepath, index=False)  # Specify index=False to avoid writing row indices to the CSV
    print(f"The csv has been successfully saved in {csv_filepath}")

    # create boxplot
    fig = iou_scores.box_plot_iou(df_concat, accuracy_txt_path=os.path.join(result_dir, 'accuracy_percentage.txt'))
    boxplot_filepath = os.path.join(result_dir, FILENAME_BOXPLOT)
    pio.write_image(fig, boxplot_filepath, format="jpg")
    print(f"The boxplot has been successfully saved in {boxplot_filepath}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time taken: {duration} seconds")
    

