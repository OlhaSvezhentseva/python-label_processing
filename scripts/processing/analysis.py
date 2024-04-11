# Import third-party libraries
import os
import argparse
import sys
import time

# Import the necessary module from the 'label_processing' module package
from label_processing.detect_empty_labels_module import find_empty_labels


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'analysis.py [-h] -o <output_image_dir> -i <input_image_dir>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the detect_empty_labels_module.py.",
        add_help=False,
        usage=usage
    )

    parser.add_argument(
        '-h', '--help',
        action='help',
        help='Description of the command-line arguments.'
    )

    parser.add_argument(
        '-o', '--output_image_dir',
        metavar='',
        type=str,
        default=os.getcwd(),
        help=('Directory where the filtered images will be stored.\n'
              'Default is the user current working directory.')
    )

    parser.add_argument(
        '-i', '--input_image_dir',
        metavar='',
        type=str,
        required=True,
        help=('Directory where the input jpgs are stored.')
    )

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()
    input_image_dir = args.input_image_dir
    output_image_dir = args.output_image_dir

    if not os.path.exists(input_image_dir):
        print(f"Error: Input directory '{input_image_dir}' not found.")
        sys.exit(1)
    elif not os.path.exists(output_image_dir):
        print(f"Error: Output directory '{output_image_dir}' not found.")
        sys.exit(1)
    else:
        find_empty_labels(input_image_dir, output_image_dir)
        print(f"\nEmpty and non-empty labels moved to respective folders in {output_image_dir}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time taken: {duration} seconds")
