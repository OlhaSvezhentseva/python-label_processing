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


def get_predicted_transcriptions(filename):
    """This function loads predictions from a json file."""
    f = open(filename)
    transcriptions = json.load(f)
    return transcriptions


def get_gold_transcriptions(filename):
    """
    This function loads gold transcriptions from a csv file.
    :param filename: filename as a string
    :return: dictionary of the form {'ID': text, }
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


def calculate_scores(gold_text, predicted_text):
    """
    This function calculates CER and WER comparing 2 strings.
    :param gold_text: gold transcription as a string
    :param predicted_text:  predicted transcription as a string
    :return: tuple of 2 scores/None
    """
    # Ignore NURIs
    if not gold_text.startswith("http") and not gold_text.startswith("MfN URI"):
        all_scores = jiwer.compute_measures(gold_text, predicted_text)
        # Calculate normalized WER
        wer = (all_scores["insertions"] + all_scores["deletions"] + all_scores["substitutions"])/len(gold_text)
        # Calculate normalized CER
        cer = calculate_cer([gold_text], [predicted_text])
        wer = round(wer, 2)
        cer = round(cer, 2)
        return wer, cer
    return None


def compare_transcriptions(gold_transcriptions, ocr_transcriptions, file_name):
    """
    This function writes evaluation results into a csv table.
    :param gold_transcriptions: ground truth data as a dictionary
    :param ocr_transcriptions: predicted transcriptions as a list of dicts
    :param file_name: the name of a CSV file which will be created
    :return: tuple of 2 lists with scores
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


def create_plot(data, name, file_name):
    """
    This function creates a violin plot and saves it.
    :param data: scores as a list
    :param name: name of the future plot
    :param file_name: the name of the file the plot will be saved in
    :return: None
    """
    plot = pd.DataFrame(data, columns=[name])
    sns.violinplot(data=plot[name], cut=1.0).set(title=name)
    # plt.savefig(f'{name}.png')
    plt.savefig(file_name)
    print(f"Plot saved in {file_name}")


def evaluate_text_predictions(ground_truth_file, predictions_file, result_folder):
    """
    This function evaluates OCR predictions
    :param ground_truth_file: ground truth data as a CSV
    :param predictions_file:  OCR output in json
    :param result_folder: name of the folder the evaluation result will be saved in
    :return: None
    """

    ground_truth = get_gold_transcriptions(ground_truth_file)
    generated_transcriptions = get_predicted_transcriptions(predictions_file)

    # create result_folder
    # parent_dir = "text_evaluation"
    # path = os.path.join(parent_dir, result_folder)
    path = os.path.join(result_folder)
    if not os.path.exists(path):
        os.mkdir(path)
    wers, cers = compare_transcriptions(ground_truth, generated_transcriptions, f"{path}/ocr_evaluation.csv")
    print(f"Mean CER: {round(np.mean(cers), 2)}, Mean WER: {round(np.mean(wers), 2)}")
    create_plot(cers, "CERs", f"{path}/cers.png")
    create_plot(wers, "WERs", f"{path}/wers.png")
    return
