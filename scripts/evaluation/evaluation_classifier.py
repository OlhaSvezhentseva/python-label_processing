# Import the necessary module from the 'label_evaluation' module package
import label_evaluation.accuracy_classifier

# Import third-party libraries
import argparse
import os
import warnings
import pandas as pd

# Suppress warning messages during execution
warnings.filterwarnings('ignore')


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'evaluation_classifier.py [-h] -o </path/to/outputs> -d </path/to/gt_dataframe>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the accuracy_classifier.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument('-h','--help',
                        action='help',
                        help='Description of the command-line arguments.')

    parser.add_argument('-o', '--out_dir',
                        metavar='',
                        type=str,
                        default=os.getcwd(),
                        help=('Directory in which the accuracy scores and plots will be stored. '
                              'Default is the current working directory.'))

    parser.add_argument('-d', '--df',
                        metavar='',
                        type=str,
                        required=True,
                        help=('Path to the input CSV file.'))

    return parser.parse_args()


if __name__ == '__main__': 
    # Parse command-line arguments
    args = parse_arguments()
    out_dir = args.out_dir

    # Read the CSV file into a DataFrame
    df = pd.read_csv(args.df, sep=';')

    # Extract 'pred' and 'gt' columns
    pred = df["pred"]
    gt = df["gt"]

    # Get unique classes from the 'gt' column
    target = df["gt"].unique().tolist()

    # 1. Accuracy Scores
    accuracy_classifier.metrics(target, pred, gt, out_dir=out_dir)
    
    # 2. Confusion Matrix
    accuracy_classifier.cm(target, pred, gt, out_dir=out_dir)
