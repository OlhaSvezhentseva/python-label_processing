#Import Librairies
import jiwer
import json
import numpy as np
import csv
import os
from cer import calculate_cer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import label_processing.utils as utils
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')


def get_predicted_transcriptions(filename: str) -> str:
    """
    Loads predictions from the OCR outputs as a json file.

        Args:
            filename (str): path to the json file

        Returns:
            transcriptions (str): loaded json file
    """
    with open(filename) as f:
        transcriptions = json.load(f)
    return transcriptions


def get_gold_transcriptions(filename: str) -> dict:
    """
    Loads predictions from the groundtruth transcriptions as a csv file.

    Args:
        filename (str): path to the groundtruth csv

    Returns:
        gold_transcriptions (dict): dictionary of the form {'ID': text}
    """
    gold_transcriptions = {}
    with open(filename, encoding='utf-8-sig') as file_in:
        next(file_in)
        for line in file_in:
            line = line.strip()
            line = line.split(';')
            if line[0] != '':
                gold_transcriptions[line[0]] = line[1]
    return gold_transcriptions


def calculate_scores(gold_text: str, predicted_text: str) -> tuple:
    """
    Calculates CER and WER by comparing the predicted and groundtruth transcriptions.

    Args:
        gold_text (str): groundtruth transcription as a string
        predicted_text (str): predicted transcription as a string

    Returns:
        wer, cer (tuple): tuple of the two scores/None
    """
    if not gold_text.startswith("http") and not gold_text.startswith("MfN URI"):
        all_scores = jiwer.compute_measures(gold_text, predicted_text)
        # Calculate normalized WER
        wer = ((all_scores["insertions"] + all_scores["deletions"] +
            all_scores["substitutions"])/len(gold_text))
        # Calculate normalized CER
        cer = calculate_cer([gold_text], [predicted_text])
        wer = round(wer, 2)
        cer = round(cer, 2)
        return wer, cer
    return None


def compare_transcriptions(gold_transcriptions: dict, ocr_transcriptions: list, file_name: str) -> tuple:
    """
    Writes evaluation results into a csv table.

    Args:
        gold_transcriptions (dict): groundtruth data as a dictionary
        ocr_transcriptions (list): predicted transcriptions as a list of dicts
        file_name (str): the name of a CSV file which will be created

    Returns:
        all_wers, all_cers (tuple): tuple of two lists with scores
    """
    all_wers = []
    all_cers = []
    f = open(f'{file_name}', 'w')
    writer = csv.writer(f)
    writer.writerow(["File ID", "Reference text", "OCR output", "WER", "CER"])
    for transcript_info in ocr_transcriptions:
        gold = gold_transcriptions[transcript_info["ID"]]
        predicted = transcript_info["text"]
        scores = calculate_scores(gold, predicted)
        if scores is not None:
            all_wers.append(scores[0])
            all_cers.append(scores[1])
            writer.writerow([transcript_info["ID"], gold, predicted, scores[0], scores[1]])
    f.close()
    return all_wers, all_cers


def create_plot(data: list, name: list, file_name: str) -> None:
    """
    Create violin plots for the CER and WER scores respectively.

    Args:
        data (list): scores as a list
        name (list): name of the future plot
        file_name (str): the name of the file the plot will be saved in
    """
    plot = pd.DataFrame(data, columns=[name])
    sns.violinplot(data=plot[name], cut=1.0).set(title=name)
    # plt.savefig(f'{name}.png')
    plt.savefig(file_name)
    print(f"Plot saved in {file_name}")


def evaluate_text_predictions(ground_truth_file: str, predictions_file: str, out_dir) -> tuple:
    """
    Evaluates OCR predictions.

    Args:
        ground_truth_file (str): path to groundtruth data as a CSV
        predictions_file (str): path to OCR output as a json file

    Returns:
        wers, cers (tuple): tuple of two lists with scores
    """
    ground_truth = get_gold_transcriptions(ground_truth_file)
    generated_transcriptions = get_predicted_transcriptions(predictions_file)
    wers, cers = compare_transcriptions(ground_truth, generated_transcriptions, f"{out_dir}/ocr_evaluation.csv")
    print(f"Mean CER: {round(np.mean(cers), 2)}, Mean WER: {round(np.mean(wers), 2)}")
    create_plot(cers, "CERs", f"{out_dir}/cers.png")
    create_plot(wers, "WERs", f"{out_dir}/wers.png")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str)
    parser.add_argument("--pred", type=str)
    parser.add_argument("--folder", nargs='?', default="result")
    args = parser.parse_args()
    evaluate_text_predictions(args.gt, args.pred, args.folder)
    
