import argparse
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
import os

def rotation_evaluation(input_csv_path, output_folder):
    """
    Create a comparison plot and save it to a given output folder.

    Args:
        input_csv_path (str): Path to the input CSV file containing data.
        output_folder (str): Path to the folder where the plot and value counts will be saved.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv_path, sep=';')

    # Group and count values
    value_counts = df.groupby(['before'])['pred'].value_counts()

    # Define conditions and choices
    conditions = [(df['pred'] == "straight"),
                 (df['pred'] == "not_straight")]
    choices = ['match', 'no_match']

    # Create a new column in DataFrame that displays results of comparisons
    df['comparison_values'] = np.select(conditions, choices, default='Tie')

    # Create a comparison plot
    sns.set_theme(style="whitegrid")
    plot = sns.displot(df, x="pred", hue="comparison_values")

    # Define the output plot file path
    output_plot_path = os.path.join(output_folder, "comparison_plot.png")

    # Save the plot to the target folder
    plot.savefig(output_plot_path)

    # Define the output text file path
    output_text_path = os.path.join(output_folder, "value_counts.txt")

    # Save the value_counts to a text file
    with open(output_text_path, 'w') as text_file:
        text_file.write(str(value_counts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and save rotation evaluation metrics.")
    parser.add_argument("input_csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")

    args = parser.parse_args()

    rotation_evaluation(args.input_csv_path, args.output_folder)
