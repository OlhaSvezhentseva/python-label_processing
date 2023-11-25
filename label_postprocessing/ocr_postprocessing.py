# Import third-party libraries
import json
import re
from nltk import word_tokenize
import string
import json
import pandas as pd

NON_ASCII = re.compile(" [^\x00-\x7F] ")
NON_ALPHA_NUM = re.compile("[^a-zA-Z\d\s]{2,}")
PIPE = re.compile("[|]")


def count_mean_token_length(tokens: str):
    """
    Calculates the mean length of tokens in a list.

    Args:
        tokens (list): List of tokens.

    Returns:
        float: Mean token length.
    """
    total_length = 0
    for token in tokens:
        total_length += len(token)
    if len(tokens) != 0:
        return total_length / len(tokens)
    return 0


def is_plausible_prediction(transcript: str):
    """
    Checks if a transcript is a plausible prediction based on the average token length.

    Args:
        transcript (str): Input transcript.

    Returns:
        bool: True if the transcript is plausible, False otherwise.
    """
    tokens = word_tokenize(transcript)
    tokens_no_punct = [token for token in tokens if token not in string.punctuation]
    average_token_length = count_mean_token_length(tokens_no_punct)
    if 0 <= average_token_length < 2:
        return False
    return True


def correct_transcript(transcript: str):
    """
    Performs corrections on a transcript, removing non-ASCII characters, multiple non-alphanumeric characters, and the pipe character.

    Args:
        transcript (str): Input transcript.

    Returns:
        str: Corrected transcript.
    """
    # remove single non-ASCII (spaces?)
    new_string = re.sub(NON_ASCII, ' ', transcript)
    # remove 2 or more non alphanumeric characters in a row
    new_string2 = re.sub(NON_ALPHA_NUM, '', new_string)
    # remove pipe character
    result = re.sub(PIPE, '', new_string2)
    return result


def is_nuri(transcript: str):
    """
    Checks if a transcript starts with "http," indicating a Nuri.

    Args:
        transcript (str): Input transcript.

    Returns:
        bool: True if the transcript is a Nuri, False otherwise.
    """
    if transcript.startswith("http"):
        return True


def is_empty(transcript: str):
    """
    Checks if a transcript is empty.

    Args:
        transcript (str): Input transcript.

    Returns:
        bool: True if the transcript is empty, False otherwise.
    """
    if len(transcript) == 0:
        return True


def save_transcripts(transcripts: dict, file_name: str):
    """
    Saves transcripts as a CSV file.

    Args:
        transcripts (dict): Dictionary of transcripts.
        file_name (str): Name of the output CSV file.
    """
    data = pd.DataFrame.from_dict(transcripts, orient="index")
    data.to_csv(file_name)


def save_json(transcripts: list, file_name: str):
    """
    Saves transcripts as a JSON file.

    Args:
        transcripts (list): List of transcripts.
        file_name (str): Name of the output JSON file.
    """
    transcripts = json.dumps(transcripts, indent=4)
    with open(file_name, "w") as outfile:
        outfile.write(transcripts)


def process_ocr_output(ocr_output: str):
    """
    Processes OCR output, categorizing and saving transcripts based on Nuri, empty, plausible, and corrected.

    Args:
        ocr_output (str): OCR output file path.
    """
    nuri_labels = {}
    empty_labels = {}
    plausible_labels = []
    clean_labels = []
    with open(ocr_output, 'r') as f:
        labels = json.load(f)
        for label in labels:
            if is_nuri(label["text"]):
                nuri_labels[label["ID"]] = label["text"]
            elif is_empty(label["text"]):
                empty_labels[label["ID"]] = ""
            elif is_plausible_prediction(label["text"]):
                plausible_label = {"ID": label["ID"], "text":label["text"]}
                plausible_labels.append(plausible_label)
                clean_transcript = correct_transcript(label["text"])
                clean_label = {"ID": label["ID"], "text": clean_transcript}
                clean_labels.append(clean_label)
    save_transcripts(nuri_labels, "nuris.csv")
    save_transcripts(empty_labels, "empty_transcripts.csv")
    save_json(plausible_labels, "plausible_transcripts.json")
    save_json(clean_labels, "corrected_transcripts.json")
    return
