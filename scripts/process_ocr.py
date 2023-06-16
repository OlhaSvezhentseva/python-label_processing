import json
import argparse
import label_processing.utils as utils
from label_processing.ocr_postprocessing import (
    is_empty,
    is_nuri,
    is_plausible_prediction,
    save_transcripts
)
"""
TODO
"""

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'process_ocr.py [-h] -j <ocr-json>'
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
            '-j', '--json-file',
            metavar='',
            type=str,
            required = True,
            help=('Path to ocr output json file')
            )

    
    args = parser.parse_args()

    return args

def main(ocr_output):
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
    utils.save_json(plausible_labels, "plausible_transcripts.json")
    utils.save_json(clean_labels, "corrected_transcripts.json")
    return

if __name__ == "__main__":
    args = parsing_args
    main(args.j)