#Import Librairies
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from pathlib import Path
import glob, os



#Accuracy Scores
def metrics(target: list, pred: pd.DataFrame, gt: pd.DataFrame)-> str:
    """
    Build a text report showing the main classification metrics,
    to measure the quality of predictions of the tensorflow classification model.

    Args:
        target (list): names matching the classes
        pred (pd.DataFrame): predicted classes
        gt (pd.DataFrame): ground truth classes
    
    Return:
        classification metrics (str): text report
    """
    print("Accuracy Score -> ",accuracy_score(pred, gt)*100)
    print(classification_report(gt, pred, target_names=target))


#Confusion Matrix
def cm (target, pred: pd.DataFrame, gt: pd.DataFrame, out_dir: Path = Path(os.getcwd()))-> plt:
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
    matrix = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target, yticklabels=target, cmap="OrRd")
    plt.ylabel('Actual')
    plt.xlabel('Predicted', labelpad=30)
    figure = matrix.get_figure()
    filename = f"{Path(out_dir).stem}_cm.png"
    cm_path = f"{out_dir}/{filename}"
    figure.savefig(cm_path)
    print(f"\nThe Confusion Matrix has been successfully saved in {out_dir}")
