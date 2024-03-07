# Import third-party libraries
import cv2
import os
import numpy as np
from glob import glob
import tensorflow as tf
from keras.models import load_model

# Define constants
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4


def rotate_image(img_path: str, angle: int, output_dir: str)-> None:
    """
    Rotate an image based on a given angle and save the rotated image.

    Args:
        img_path (str): Path to the input image file.
        angle (int): Angle of rotation in multiples of 90 degrees.
        output_dir (str): Directory where the rotated image will be saved.

    Returns:
        None
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
        target_angle = (4 - angle) % NUM_CLASSES  # Calculate the required rotation to reach 0 degree

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

        # Construct the output file path
        output_dir = os.path.join(output_dir, os.path.basename(img_path))

        # Write the rotated image back to the file
        success = cv2.imwrite(img_path, rotated_img)
        if not success:
            print(f"Error: Failed to write rotated image '{img_path}' to file.")
            return

        print(f"Successfully rotated image '{img_path}' by {target_angle * 90} degrees to reach 0 degree.")
    except Exception as e:
        print(f"Error: An exception occurred while processing image '{img_path}': {e}")


def predict_angles(input_image_dir: str, output_image_dir: str) -> None:
    """
    Load a trained model, predict angles for input images, and rotate images accordingly.

    Args:
        input_image_dir (str): Directory containing input images.
        output_image_dir (str): Directory to save rotated images.

    Returns:
        None
    """
    # Load images and labels
    loaded_images = []
    for img_path in glob(os.path.join(input_image_dir, '*.jpg')):
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMAGE_SIZE)
        loaded_images.append(img)
        
    # Load the trained model
    model_path = '../../models/rotation_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Load the model
    model = load_model(model_path)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    images = np.array(loaded_images)

    # Predict using the model
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)

    # List all image files in the data directory
    filenames = [os.path.join(input_image_dir, filename) for filename in os.listdir(input_image_dir) if filename.endswith('.jpg')]

    # Apply rotation to images based on their predicted angles
    for img_path, predicted_angle in zip(filenames, predicted_labels):
        # Save rotated image to the output directory
        rotate_image(img_path, predicted_angle, output_image_dir)
