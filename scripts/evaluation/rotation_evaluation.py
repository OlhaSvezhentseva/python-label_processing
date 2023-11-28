# Import third-party libraries
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import os

def rotation_evaluation(input_csv_path: str, output_folder: str) -> None:
    """
    Create a comparison plot and save it to a given output folder.

    Args:
        input_csv_path (str): Path to the input CSV file containing data.
        output_folder (str): Path to the folder where the plot and value counts will be saved.
    """
    df = pd.read_csv(input_csv_path, sep=';')
    value_counts = df.groupby(['before'])['pred'].value_counts()
    conditions = [(df['pred'] == "straight"),
                 (df['pred'] == "not_straight")]
    choices = ['match', 'no_match']
    df['comparison_values'] = np.select(conditions, choices, default='Tie')
    sns.set_theme(style="whitegrid")
    plot = sns.displot(df, x="pred", hue="comparison_values")
    output_plot_path = os.path.join(output_folder, "comparison_plot.png")
    plot.savefig(output_plot_path)
    output_text_path = os.path.join(output_folder, "value_counts.txt")
    with open(output_text_path, 'w') as text_file:
        text_file.write(str(value_counts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and save rotation evaluation metrics.")
    parser.add_argument("input_csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")

    args = parser.parse_args()

    rotation_evaluation(args.input_csv_path, args.output_folder)
