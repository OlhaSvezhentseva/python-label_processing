#Import Librairies
import json
from nltk import word_tokenize
import pandas as pd
import string


def contains_only_letters(token: str) -> bool:
    """
    The function checks if a token consists only of letters

    Args:
        token (str): token from work_tokenize

    Returns:
        bool: True if token contains only letters
    """
    for letter in token:
        if not letter.isalpha():
            return False
    return True


def is_punctuation(token: str) -> bool:
    """
    Check if a token is a punctuation mark.

    Args:
        token (str): The token to check for punctuation.

    Returns:
        bool: True if the token is a punctuation mark, False otherwise.
    """
    if token in string.punctuation:
        return True
    return False


def extract_vocabulary(ocr_output: str) -> None:
    """
    The function extracts unique words from the transcripts.
    These words must solely contain letters and be at least 3 characters long.
    
    Args:
        ocr_output (str): ocr output
    """
    
    vocabulary = {}
    with open(ocr_output, 'r') as f:
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
