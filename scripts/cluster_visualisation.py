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

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'crop_seg.py [-h] [-c N] [-m <model/number>] [-np N]\
    -j </path/to/jpgs> -o </path/to/jpgs_outputs> '
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
            ),
    parser.add_argument(
            '-gt', '--ground_truth',
            metavar='',
            type=str,
            required=True,
            help=('path to ground truth json')
            )
    args = parser.parse_args()

    return args
    

def is_word(token):
    if token not in string.punctuation and not token.isspace():
        if len(token) >= 3:
            return True 

def build_word_vectors(labels):
    """The function builds a vector for each word in the label."""
    tokenized_labels = []
    for label in labels:
        tokens = [token.lower() for token in word_tokenize(label["text"]) if is_word(token)]
        tokenized_label = {"ID": label["ID"], "tokens": tokens}
        tokenized_labels.append(tokenized_label)
    model = gensim.models.Word2Vec([label["tokens"] for label in tokenized_labels], min_count = 1, vector_size = 100,
                                             window = 2, sg = 1)
    return model,  tokenized_labels


def build_mean_label_vector(model, labels):
    """The function builds a vector for a label (by taking the mean of word vectors)."""
    labels_vectors = {}
    for label in labels:
        # np.mean([model.wv[token] for token in label["tokens"]])
        mean_vector = np.mean([model.wv[token] for token in label["tokens"]], axis=0)
        labels_vectors[label["ID"]] = mean_vector
    return labels_vectors

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def main(ground_truth_json: str, cluster_json: str, out_dir: str):
    labels_file = ground_truth_json
    labels = load_json(labels_file)
    model1, tokens = build_word_vectors(labels)
    label_vectors = build_mean_label_vector(model1, tokens)
    clusters = load_json(cluster_json)
    clusters_sorted = [clusters[file_id][0] for file_id in label_vectors]
    data = np.array(list(label_vectors.values()))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    df['y'] = clusters_sorted 
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        style = "y",
        palette=sns.color_palette("hls", len(set(df.y))),
        data=df,
        legend="full",
        alpha=0.9
    ).set(title="Label data T-SNE projection")
    plt.legend(loc = 'upper right', bbox_to_anchor=(1.15, 1.5), ncol=1)
    plt.savefig(os.path.join(out_dir, "cluster_plot.jpg"))
    return 0
    
if __name__ == "__main__":
    args = parsing_args()
    exit(main(args.ground_truth, args.cluster_json, args.out_dir))