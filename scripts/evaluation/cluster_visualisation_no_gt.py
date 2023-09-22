"""
Module creating and saving a visual (scatter plot) of the clustering´s results.
"""

import gensim
import json
import string
import argparse
import os

from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'cluster_visualisation.py [-h] [-c N] \
    -gt <ground truth ocr output> -c <cluster output>  -o <path to output directory>'
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
            '-o', '--out_dir',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the resulting crops and the csv will be stored.\n'
                  'Default is the user current working directory.')
            ),
    parser.add_argument(
            '-c', '--cluster_json',
            metavar='',
            type=str,
            required=True,
            help=('path to cluster_json')
            )

    args = parser.parse_args()

    return args
    

def is_word(token: str):
    """
    Check if a token is a valid word.

    Args:
        token (str): The token to check.

    Returns:
        bool: True if the token is a valid word, False otherwise.
    """
    if token not in string.punctuation and not token.isspace():
        if len(token) >= 3:
            return True 

def build_word_vectors(labels):
    """
    Build word vectors for labels.

    Args:
        labels (list): List of label objects containing text.

    Returns:
        gensim.models.Word2Vec: Word2Vec model trained on label tokens.
        list: List of tokenized labels with associated IDs.
    """
    tokenized_labels = []
    for label in labels.keys():
        tokens = [token.lower() for token in word_tokenize(labels[label][1]) if is_word(token)]
        if len(tokens) >0:
            tokenized_label = {"ID": label, "tokens": tokens}
            tokenized_labels.append(tokenized_label)
    model = gensim.models.Word2Vec([label["tokens"] for label in tokenized_labels], min_count = 1, vector_size = 100,
                                             window = 2, sg = 1)
    return model,  tokenized_labels


def build_mean_label_vector(model: gensim.models.Word2Vec, labels: list):
    """
    Build a vector for a label by taking the mean of word vectors.

    Args:
        model (gensim.models.Word2Vec): Word2Vec model.
        labels (list): List of tokenized labels with associated IDs.

    Returns:
        dict: Dictionary mapping label IDs to their mean vectors.
    """
    labels_vectors = {}
    for label in labels:
        # np.mean([model.wv[token] for token in label["tokens"]])
        mean_vector = np.mean([model.wv[token] for token in label["tokens"]], axis=0)
        labels_vectors[label["ID"]] = mean_vector
    return labels_vectors

def load_json(file: str):
    """
    Load data from a JSON file.

    Args:
        file (str): Path to the JSON file.

    Returns:
        dict: Loaded JSON data.
    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def main(cluster_json: str, out_dir: str):
    """
    Main function for processing label data, performing T-SNE dimensionality reduction, and saving a scatter plot.

    Args:
        ground_truth_json (str): Path to the ground truth JSON file.
        cluster_json (str): Path to the cluster JSON file.
        out_dir (str): Directory where the scatter plot image will be saved.
    """
    labels_file = cluster_json
    labels = load_json(labels_file)
    model1, tokens = build_word_vectors(labels)
    label_vectors = build_mean_label_vector(model1, tokens)
    
    clusters_sorted = [labels[file_id][0] for file_id in label_vectors]
    data = np.array(list(label_vectors.values()))
    #data = list(label_vectors.values())
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    df['y'] = clusters_sorted 
    
    plt.figure(figsize=(20, 12))  # Increase the figure size
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        style="y",
        palette=sns.color_palette("hls", len(set(df.y))),
        data=df,
        legend="full",
        alpha=0.9
    ).set(title="Label data T-SNE projection")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.5), ncol=1)
    plt.tight_layout()  # Adjust margins
    plt.savefig(os.path.join(out_dir, "cluster_plot.png"), bbox_inches='tight')  # Save with bbox_inches='tight'
    return print(f"\nThe image has been successfully saved in {out_dir}")

    
if __name__ == "__main__":
    args = parsing_args()
    exit(main(args.cluster_json, args.out_dir))
