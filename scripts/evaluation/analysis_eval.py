# Import third-party libraries
import os
import argparse
import sys
import time


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'analysis_eval.py [-h] -e <empty_folder> -n <not_empty_folder>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the evaluation_detect_empty_labels.py.",
        add_help=False,
        usage=usage
    )

    parser.add_argument(
        '-h', '--help',
        action='help',
        help='Description of the command-line arguments.'
    )

    parser.add_argument(
        '-e', '--empty_folder',
        metavar='',
        type=str,
        required=True,
        help=('Directory where the predicted empty labels images are stored.')
    )

    parser.add_argument(
        '-n', '--not_empty_folder',
        metavar='',
        type=str,
        required=True,
        help=('Directory where the predicted not_empty labels images are stored.')
    )

    return parser.parse_args()


def evaluate_labels(empty_folder: str, not_empty_folder: str) -> None:
    """
    Evaluate the predicted labels against the ground truth labels.

    Args:
        empty_folder (str): Path to the directory containing predicted empty labels images.
        not_empty_folder (str): Path to the directory containing predicted not empty_labels images.
    """
    correct_empty = 0
    total_empty = 0
    for filename in os.listdir(empty_folder):
        total_empty += 1
        label = filename.split("__")[-1].split(".")[0]
        if label == "empty":
            correct_empty += 1
    
    correct_not_empty = 0
    total_not_empty = 0
    for filename in os.listdir(not_empty_folder):
        total_not_empty += 1
        label = filename.split("__")[-1].split(".")[0]
        if label != "empty":
            correct_not_empty += 1
    
    empty_accuracy = correct_empty / total_empty if total_empty != 0 else 0
    not_empty_accuracy = correct_not_empty / total_not_empty if total_not_empty != 0 else 0

    total_correct = correct_empty + correct_not_empty
    total_files = total_empty + total_not_empty
    total_accuracy = total_correct / total_files if total_files != 0 else 0

    # Print evaluation metrics
    print(f"Empty folder accuracy: {empty_accuracy:.2%} ({correct_empty}/{total_empty})")
    print(f"Not empty folder accuracy: {not_empty_accuracy:.2%} ({correct_not_empty}/{total_not_empty})")
    print(f"Total accuracy: {total_accuracy:.2%} ({total_correct}/{total_files})")
    

if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()
    empty_image_dir = args.empty_folder
    not_empty_image_dir = args.not_empty_folder

    if not os.path.exists(empty_image_dir):
        print(f"Error: Input directory '{empty_image_dir}' not found.")
        sys.exit(1)
    elif not os.path.exists(not_empty_image_dir):
        print(f"Error: Input directory '{not_empty_image_dir}' not found.")
        sys.exit(1)
    else:
        evaluate_labels(empty_image_dir, not_empty_image_dir)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time taken: {duration} seconds")
