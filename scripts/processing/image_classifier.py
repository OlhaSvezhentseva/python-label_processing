# Import the necessary module from the 'label_processing' module package
import label_processing.tensorflow_classifier

# Import third-party libraries
import argparse
import os
import warnings

# Suppress warning messages during execution
warnings.filterwarnings('ignore')


def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'python image_classifier.py [-h] -m <model_number> -j <path_to_jpgs> -o <path_to_outputs>'
    
    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Execute the tensorflow_classifier.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )

    parser.add_argument(
        '-m', '--model',
        type=int,
        choices=range(1, 4),
        help=('Write the number of the selected classifier model:\n'
             '1: nuri or not_nuri\n'
             '2: handwritten or printed\n'
             '3: multi label image or single label image')
    )

    parser.add_argument(
        '-o', '--out_dir',
        type=str,
        default=os.getcwd(),
        help=('Directory to store outputs: classified pictures and CSV (default: current working directory).')
    )

    parser.add_argument(
        '-j', '--jpg_dir',
        type=str,
        required=True,
        help=('Directory where the inputs (JPEG images) are stored.')
    )

    return parser.parse_args()

def get_model_path(model_int):
    """
    Get the path to the selected model based on the model integer.

    Args:
        model_int (int): Integer for model selection.

    Returns:
        str: Path to the selected model file.
    """
    script_dir = os.path.dirname(__file__)
    model_paths = {
        1: "../../models/model_classifier_nuri_not_nuri",
        2: "../../models/model_classifier_hp",
        3: "../../models/model_classifier_multi_single"
    }
    return os.path.abspath(os.path.join(script_dir, model_paths.get(model_int)))


def get_class_names(model_int):
    """
    Get the class names based on the model integer.

    Args:
        model_int (int): Integer for class selection.

    Returns:
        list: List with the selected classes.
    """
    class_names = {
        1: ["not_nuri", "nuri"],
        2: ["handwritten", "printed"],
        3: ["multi", "single"]
    }
    return class_names.get(model_int)


def main():
    """
    Main function to execute the script.
    """
    args = parse_arguments()
    
    model_path = get_model_path(args.model)
    class_names = get_class_names(args.model)
    jpeg_dir = args.jpg_dir
    out_dir = args.out_dir

    # Call the Model
    model = tensorflow_classifier.get_model(model_path)

    # Model Predictions and save CSV
    df = tensorflow_classifier.class_prediction(model, class_names, jpeg_dir, out_dir=out_dir)

    # Save classified pictures
    tensorflow_classifier.filter_pictures(jpeg_dir, df, out_dir=out_dir)

if __name__ == '__main__':
    main()
