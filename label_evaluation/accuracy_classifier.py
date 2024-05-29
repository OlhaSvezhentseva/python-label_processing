# Import third-party libraries
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from pathlib import Path
import os


# Accuracy Scores
def metrics(target: list, pred: pd.DataFrame, gt: pd.DataFrame, out_dir: Path = Path(os.getcwd())) -> str:
    """
    Build a text report showing the main classification metrics,
    to measure the quality of predictions of the tensorflow classification model, and save it to a text file.

    Args:
        target (list): names matching the classes
        pred (pd.DataFrame): predicted classes
        gt (pd.DataFrame): ground truth classes
        out_dir (Path): Directory where the report file will be saved

    Return:
        classification metrics (str): text report
    """
    report_file = os.path.join(out_dir, "classification_report.txt")
    
    accuracy = accuracy_score(pred, gt) * 100
    report = classification_report(gt, pred, target_names=target)

    # Print accuracy to console
    print("Accuracy Score -> ", accuracy)

    # Print classification report to console
    print(report)

    # Save the classification report to a text file
    with open(report_file, 'w') as file:
        file.write(f"Accuracy Score -> {accuracy}\n")
        file.write(report)

    print(f"\nThe Classification Report has been successfully saved in {out_dir}")

    return report


# Confusion Matrix
def cm (target: list, pred: pd.DataFrame, gt: pd.DataFrame, out_dir: Path = Path(os.getcwd()))-> plt:
    """
    Compute confusion matrix to evaluate the performance of the classification.

    Args:
        target (list): names matching the classes
        pred (pd.DataFrame): predicted classes
        gt (pd.DataFrame): ground truth classes
        out_dir (Path): path to the target directory to save the confusion matrix plot.
    
    Return:
        confusion matrix (plt): confusion matrix as a heatmap
    """
    cm = confusion_matrix(gt, pred)

    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(15,10))
    matrix = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target, yticklabels=target, cmap="OrRd",
                        annot_kws={"size": 14})
    plt.ylabel('Ground truth', fontsize=18)
    plt.xlabel('Predictions', labelpad=30, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    figure = matrix.get_figure()
    filename = f"{Path(out_dir).stem}_cm.png"
    cm_path = f"{out_dir}/{filename}"
    figure.savefig(cm_path)
    
    print(f"\nThe Confusion Matrix has been successfully saved in {out_dir}")
