#Import Librairies
import json
from nltk import word_tokenize
import pandas as pd
import string


def contains_only_letters(token: str):
    """The function checks if a token consists only of letters."""
    for letter in token:
        if not letter.isalpha():
            return False
    return True


def is_punctuation(token: str):
    """The function checks if a token is a punctuation mark."""
    if token in string.punctuation:
        return True


def extract_vocabulary(ocr):
    """
    The function extracts unique words from the transcripts.
    These words must solely contain letters and be at least 3 characters long.
    """
    vocabulary = {}
    with open(ocr, 'r') as f:
        labels = json.load(f)
        for label in labels:
            tokens = word_tokenize(label["text"])
            for token in tokens:
                token = token.lower()
                if is_punctuation(token):
                    pass
                elif len(token) >= 3:
                    if contains_only_letters(token):
                        if token in vocabulary:
                            vocabulary[token] += 1
                        else:
                            vocabulary[token] = 1
    # df = pd.DataFrame.from_dict(vocabulary, orient='index')
    df = pd.DataFrame(vocabulary.items(), columns=['Type', 'Count'])
    new_df = df.sort_values(by=['Count'], ascending=False)
    new_df.to_csv("vocabulary.csv", index=False)
    return "Vocabulary saved"
