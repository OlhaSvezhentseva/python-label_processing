# Import third-party libraries
import json
import os
import argparse
import time

# Import the necessary module from the 'label_processing' and `label_postprocessing` module packages
import label_processing.utils as utils
from label_postprocessing.ocr_postprocessing import (
    is_empty,
    is_nuri,
    is_plausible_prediction,
    save_transcripts,
    correct_transcript
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'process.py [-h] -j <ocr-json> -o <out-dir>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the ocr_postprocessing.py module.",
        add_help = False,
        usage = usage)

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
            help=('Path to ocr output json file.')
            )

    parser.add_argument(
            '-o', '--outdir',
            metavar='',
            type=str,
            required = True,
            help=('Output directory where files should be saved.')
            )

    return parser.parse_args()


def main(ocr_output: str, outdir: str) -> None:
    """
    Process OCR output and perform various tasks like identifying Nuri labels, empty labels, and correcting plausible labels.

    Args:
        ocr_output (str): The path to the OCR output JSON file.
        outdir (str): The directory where the output files will be saved.
    """
    start_time = time.time()
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
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time taken: {duration} seconds")
    return 0

if __name__ == "__main__":
    args = parse_arguments()
    exit(main(args.json, args.outdir))
