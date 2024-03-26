# Third-Party Libraries
import os
from glob import glob
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.models import load_model
import time

# Define constants
IMAGE_SIZE = (224, 224)
TEXT_FILE = "accuracy_metrics.txt"
ANGLE_NAMES = {0: '0', 1: '90', 2: '180', 3: '270'}

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

        true_labels = np.append(true_labels, angle)  # Append the class label to true_labels

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
    plt.xticks(ticks=np.arange(4) + 0.5, labels=[ANGLE_NAMES[i] for i in range(4)])
    plt.yticks(ticks=np.arange(4) + 0.5, labels=[ANGLE_NAMES[i] for i in range(4)])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(confusion_matrix_plot_path)
    plt.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and save rotation evaluation metrics.")
    parser.add_argument("input_image_dir", type=str, help="Path to the image input folder.")
    parser.add_argument("output_folder_path", type=str, help="Path to the output folder.")

    args = parser.parse_args()
    start_time = time.time()

    rotation_evaluation(args.input_image_dir, args.output_folder_path)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time taken: {duration} seconds")
