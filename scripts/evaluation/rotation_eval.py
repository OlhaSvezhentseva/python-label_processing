# Third-Party Libraries
import argparse
import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.models import load_model
from keras.layers import BatchNormalization
import time

# Define constants
IMAGE_SIZE = (224, 224)
TEXT_FILE = "accuracy_metrics.txt"
ANGLE_NAMES = ['0', '90', '180', '270']
NUM_CLASSES = 4


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'rotation_eval.py [-h] -i <input image dir> -o <output folder path>'

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(
        description="Create and save rotation evaluation metrics.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-i', '--input_image_dir',
            metavar='',
            type=str,
            required = True,
            help=('Path to the image input folder.')
            )
            
    parser.add_argument(
            '-o', '--output_folder_path',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Path to the output folder.')
            )

    return parser.parse_args()


def rotate_image(img_path: str , angle: int) -> None:
    """
    Rotates an image by the specified angle and saves it back to the same path.

    Args:
        img_path (str): Path to the image file.
        angle (int): Angle in multiples of 90 degrees to rotate the image.
                     Valid values are [0, 1, 2, 3], corresponding to [0, 90, 180, 270] degrees.

    Returns:
        None. The rotated image is saved back to the file specified by img_path.

    Raises:
        Exception: If there is an error reading or writing the image file, or performing the rotation.
    """
    try:
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image '{img_path}'.")
            return

        # Check if the angle is not 0
        if angle == 0:
            print(f"Skipping image '{img_path}' as it does not need rotation.")
            return

        # Get image dimensions
        height, width = img.shape[:2]

        # Calculate the target angle to rotate the image
        target_angle = (4 - angle) % NUM_CLASSES  # Calculate the required rotation to reach 0 degrees

        # Rotate the image around its center
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), target_angle * 90, 1)

        # Compute the new dimensions of the rotated image
        cos_theta = np.abs(rotation_matrix[0, 0])
        sin_theta = np.abs(rotation_matrix[0, 1])
        new_width = int(height * sin_theta + width * cos_theta)
        new_height = int(height * cos_theta + width * sin_theta)

        # Adjust the rotation matrix to take into account the translation
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2

        # Perform the rotation with an enlarged canvas
        rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))

        # Write the rotated image back to the file
        success = cv2.imwrite(img_path, rotated_img)
        if not success:
            print(f"Error: Failed to write rotated image '{img_path}' to file.")
            return

        print(f"Successfully rotated image '{img_path}' by {target_angle * 90} degrees to reach 0 degree.")
    except Exception as e:
        print(f"Error: An exception occurred while processing image '{img_path}': {e}")


def rotation_evaluation(input_image_dir: str, output_folder_path: str) -> None:
    """
    Load a trained model, predict angles for input images, and evaluate the prediction's accuracy.

    Args:
        input_image_dir (str): Directory containing input images.
        output_folder_path (str): Path to the folder to save results.

    Returns:
        None
    """
    true_labels = np.array([])
    loaded_images = []
    for img_path in glob(os.path.join(input_image_dir, '*.jpg')):
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMAGE_SIZE)
        loaded_images.append(img)

        # Extract angle information from the file name - Ground Truth
        angle_str = img_path.split('__')[-1].split('.')[0]
        angle = int(angle_str) // 90

        true_labels = np.append(true_labels, angle)

    test_images = np.array(loaded_images)

    # Load the trained model
    model_path = '../../models/rotation_model.h5'
    custom_objects = {"BatchNormalization": BatchNormalization}
    try:
        model = load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Predict using the model
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # List all image files in the test data directory
    filenames = [os.path.join(input_image_dir, filename) for filename in os.listdir(input_image_dir) if filename.endswith('.jpg')]

    # Apply rotation to images based on their predicted angles
    for img_path, predicted_angle in zip(filenames, predicted_labels):
        rotate_image(img_path, predicted_angle)

    # Evaluate predictions
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    # Compute class weights and weighted evaluation metrics
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(true_labels), y=true_labels)
    weighted_accuracy = np.sum(np.diag(conf_matrix) / np.sum(conf_matrix, axis=1) * class_weights)
    weighted_precision = np.sum(np.diag(conf_matrix) / np.sum(conf_matrix, axis=0) * class_weights)
    weighted_recall = np.sum(np.diag(conf_matrix) / np.sum(conf_matrix, axis=1) * class_weights)
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)

    # Save accuracy metrics to text file
    accuracy_file_path = os.path.join(output_folder_path, TEXT_FILE)
    with open(accuracy_file_path, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1-score: {f1:.2f}\n\n")
        f.write("Weighted Evaluation Metrics:\n")
        f.write(f"Weighted Accuracy: {weighted_accuracy:.2f}\n")
        f.write(f"Weighted Precision: {weighted_precision:.2f}\n")
        f.write(f"Weighted Recall: {weighted_recall:.2f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.2f}\n\n")

    # Save confusion matrix plot as PNG
    confusion_matrix_plot_path = os.path.join(output_folder_path, 'confusion_matrix.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xticks(ticks=np.arange(4) + 0.5, labels=ANGLE_NAMES)
    plt.yticks(ticks=np.arange(4) + 0.5, labels=ANGLE_NAMES)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Label Rotation confusion matrix with misclassification count', fontsize=20, pad=20)
    plt.savefig(confusion_matrix_plot_path)
    plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    input = args.input_image_dir
    output = args.output_folder_path
    start_time = time.time()

    rotation_evaluation(input, output)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time taken: {duration} seconds")
