"""
Responsible for filtering the OCR ouputs according to 4 categories:nuris, empty transcripts, plausible output, nonsense output.
Plausible outputs are corrected using regular expressions and is saved as corrected_transcripts.json.
"""
import json
import os
import argparse
import label_processing.utils as utils
from label_postprocessing.ocr_postprocessing import (
    is_empty,
    is_nuri,
    is_plausible_prediction,
    save_transcripts,
    correct_transcript
)

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
            '-j', '--json',
            metavar='',
            type=str,
            required = True,
            help=('Path to ocr output json file')
            )

    parser.add_argument(
            '-o', '--outdir',
            metavar='',
            type=str,
            required = True,
            help=('output directory where files should be saved')
            )

    
    args = parser.parse_args()

    return args

def main(ocr_output: str, outdir: str) -> None:
    """
    Process OCR output and perform various tasks like identifying Nuri labels, empty labels, and correcting plausible labels.

    Args:
        ocr_output (str): The path to the OCR output JSON file.
        outdir (str): The directory where the output files will be saved.
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
    save_transcripts(nuri_labels, os.path.join(outdir, "nuris.csv"))
    save_transcripts(empty_labels, os.path.join(outdir, "empty_transcripts.csv"))
    utils.save_json(plausible_labels, "plausible_transcripts.json", outdir)
    utils.save_json(clean_labels, "corrected_transcripts.json", outdir)
    return 0

if __name__ == "__main__":
    args = parsing_args()
    exit(main(args.json, args.outdir))
