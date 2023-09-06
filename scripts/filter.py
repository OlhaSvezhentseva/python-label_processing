#Import Librairies
import re
from nltk import word_tokenize
import string
import json
import argparse
import pandas as pd
#Import module from this package
from label_postprocessing.utils import dump_json


NON_ASCII = re.compile(" [^\x00-\x7F] ")
NON_ALPHA_NUM = re.compile("[^a-zA-Z\d\s]{2,}")
PIPE = re.compile("[|]")


def count_mean_token_length(tokens: list[str]) -> int|float:
    """
    The function counts mean token length in the list of tokens.

    Args:
        tokens (list): list of tokens

    Returns:
        int|float: mean token length
    """
    total_length = 0
    for token in tokens:
        total_length += len(token)
    if len(tokens) != 0:
        return total_length / len(tokens)
    return 0


def is_plausible_prediction(transcript: str) -> bool:
    """
    The function check if the transcript seems to be plausible.

    Args:
        transcript (str): transcript string

    Returns:
        bool: true if transcript is predicted to be plausible or not
    """
    tokens = word_tokenize(transcript)
    tokens_no_punct = [token for token in tokens if token not in string.punctuation]
    average_token_length = count_mean_token_length(tokens_no_punct)
    if 0 <= average_token_length < 3:
        return False
    return True


def correct_transcript(transcript: str) -> str:
    """
    The function corrects the transcript.

    Args:
        transcript (str): raw transcript

    Returns:
        str: corrected transcript
    """      
    # remove single non-ASCII (spaces?)
    new_string = re.sub(NON_ASCII, ' ', transcript)
    # remove 2 or more non alphanumeric characters in a row
    new_string2 = re.sub(NON_ALPHA_NUM, '', new_string)
    # remove pipe character
    result = re.sub(PIPE, '', new_string2)
    return result


def is_nuri(transcript: str) -> bool:
    """The function checks if the transcript is a nuri."""
    if transcript.startswith("http"):
        return True
    return False


def is_empty(transcript: str) -> bool:
    """The function checks if the transcript is empty."""
    if len(transcript) == 0:
        return True
    return False


def save_transcripts(transcripts: dict, file_name: str) -> None:
    """The function saves transcripts as a csv-file."""
    data = pd.DataFrame.from_dict(transcripts, orient="index")
    data.to_csv(file_name)


def process_ocr_output(ocr_output: str) -> str:
    """
    The function assigns transcripts to a corresponding category and cleans it if necessary.

    Args:
        ocr_output (str): path to ocr output
    """
    nuri_labels = {}
    empty_labels = {}
    implausible_labels = {}
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
            else:
                implausible_labels[label["ID"]] = label["text"]
    save_transcripts(nuri_labels, "nuris.csv")
    save_transcripts(empty_labels, "empty_transcripts.csv")
    save_transcripts(implausible_labels, "nonsense_transcripts.csv")
    dump_json(plausible_labels, "plausible_transcripts.json")
    dump_json(clean_labels, "corrected_transcripts.json")
    return f"Data was filtered."


# print(process_ocr_output("ocr_pytesseract_all.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr_output", type=str)
    args = parser.parse_args()
    print(process_ocr_output(args.ocr_output))


