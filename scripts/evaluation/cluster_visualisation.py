"""
Module creating and saving a visual (scatter plot) of the clustering´s results.
"""

import gensim
import json
import string
import argparse
import os

import plotly.express as px  #import plotly express

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
    -gt <ground truth ocr output> -c <cluster output>  -o <path to output directory> -s <cluster_size>'
    parser = argparse.ArgumentParser(description=__doc__,
                                     add_help=False,
                                     usage=usage
                                     )

    parser.add_argument(
        '-h', '--help',
        action='help',
        help='Open this help text.'
    )
    parser.add_argument(
        '-o', '--out_dir',
        metavar='',
        type=str,
        default=os.getcwd(),
        help=('Directory in which the resulting crops and the csv will be stored.\n'
              'Default is the user current working directory.')
    ),
    parser.add_argument(
        '-c', '--cluster_json',
        metavar='',
        type=str,
        required=True,
        help=('path to cluster_json')
    ),
    parser.add_argument(
        '-gt', '--ground_truth',
        metavar='',
        type=str,
        default=None,
        help=('path to ground truth json')
    )

    parser.add_argument(
        '-s', '--cluster_size',
        metavar='',
        type=str,
        default = 1,
        help=('Minimal number of labels the cluster must have to be plotted')
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


def build_word_vectors(labels, ground_truth=True):
    """
    Build word vectors for labels.

    Args:
        labels (list): List of label objects containing text.

    Returns:
        gensim.models.Word2Vec: Word2Vec model trained on label tokens.
        list: List of tokenized labels with associated IDs.
    """
    tokenized_labels = []
    if ground_truth:
        for label in labels:
            tokens = [token.lower() for token in word_tokenize(label["text"]) if is_word(token)]
            tokenized_label = {"ID": label["ID"], "tokens": tokens}
            tokenized_labels.append(tokenized_label)
    else:
        print('Not gr')
        for label in labels.keys():
            tokens = [token.lower() for token in word_tokenize(labels[label][1]) if is_word(token)]
            if len(tokens) > 0:
                tokenized_label = {"ID": label, "tokens": tokens}
                tokenized_labels.append(tokenized_label)

    model = gensim.models.Word2Vec([label["tokens"] for label in tokenized_labels], min_count=1, vector_size=100,
                                   window=2, sg=1)
    return model, tokenized_labels


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


def count_cluster_size(vectors, all_labels):
    cluster_counts = {}
    for file_id in vectors:
        cluster = all_labels[file_id][0]
        if cluster in cluster_counts:
            cluster_counts[cluster] += 1
        else:
            cluster_counts[cluster] = 1
    return cluster_counts


def main(ground_truth: str, clusters_file: str, out_dir: str, cluster_size: str):
    """
    Main function for processing label data, performing T-SNE dimensionality reduction, and saving a scatter plot.

    Args:
        ground_truth (str): Path to the ground truth JSON file.
        clusters_file (str): Path to the cluster JSON file.
        out_dir (str): Directory where the scatter plot image will be saved.
        cluster_size str): The minimal size of cluster that will be plotted

    """
    if ground_truth:
        labels = load_json(ground_truth)
        clusters = load_json(clusters_file)
        model1, tokens = build_word_vectors(labels, ground_truth=True)

    else:
        labels = load_json(clusters_file)
        print(labels)
        clusters = labels
        model1, tokens = build_word_vectors(labels, ground_truth=False)
        # print(tokens)

    label_vectors = build_mean_label_vector(model1, tokens)
    cluster_counts = count_cluster_size(label_vectors, labels)
    clusters_to_plot = [cluster for cluster, count in cluster_counts.items() if count >= int(cluster_size)]

    # filter label vectors based on clusters to plot
    filtered_label_vectors = {file_id: vector for file_id, vector in label_vectors.items() if
                              labels[file_id][0] in clusters_to_plot}
    clusters_sorted = [labels[file_id][0] for file_id in filtered_label_vectors]
    label_ids = [file_id for file_id in filtered_label_vectors]  # extract label IDs
    tokens_list = [labels[file_id][1] for file_id in filtered_label_vectors]  # extract tokens
    data = np.array(list(filtered_label_vectors.values()))


    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['y'] = clusters_sorted
    df['label_id'] = label_ids  # add label IDs to df
    df['tokens'] = tokens_list  # add tokens to df

    # create an interactive scatter plot using plotly express
    fig = px.scatter(
        df, x="tsne-2d-one", y="tsne-2d-two", color="y",
        hover_name="label_id",
        hover_data=["tokens"],
    )

    fig.update_layout(
        title="Label Data T-SNE projection",
        legend=dict(x=1.15, y=1.5),
    )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    fig.write_html(os.path.join(out_dir, "cluster_plot.html"))
    return print(f"\nThe interactive scatter plot has been successfully saved in {out_dir}")

    '''
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
    '''


if __name__ == "__main__":
    args = parsing_args()
    exit(main(args.ground_truth, args.cluster_json, args.out_dir, args.cluster_size))
