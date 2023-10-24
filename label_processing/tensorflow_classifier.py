#Import Librairies
import numpy as np
import pandas as pd
import cv2
import glob, os
import pathlib
from pathlib import Path
import re

import tensorflow as tf
from tensorflow import keras

from label_processing import utils


#--------------------------------Predict Classes--------------------------------#


def get_model(path_to_model: str) -> tf.keras.Sequential:
    """
    Call trained Keras Sequential image classifier model.
    The model was trained with Tensorflow.
    
    Args:
        path_to_model (str): string that contains the path to the model.
            
    Returns:
        model (tf.keras.Sequential): trained Keras Sequential image classifier model.
    """
    print("\nCalling classification model")
    model = tf.keras.models.load_model(path_to_model)
    return model


def class_prediction(model: tf.keras.Sequential, class_names: list, jpg_dir: str, out_dir = None) -> pd.DataFrame:
    """
    Creates a dataframe only with predicted class for each picture.

    Args:
        model (tf.keras.Sequential): trained Keras Sequential image classifier model.
        class_names (list): model's predicted classes
        jpg_dir (str): string with path to the directory containing the
                           original jpgs.

    Returns:
        DataFrame (pd.DataFrame): Pandas Dataframe with the predicted results.
    """
    utils.check_dir(jpg_dir)
    print("\nPredicting classes")
    all_predictions =[]
    img_width = 180
    img_height = 180
    for file in glob.glob(f"{jpg_dir}/*.jpg"):
        image = tf.keras.utils.load_img(file, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        entry = {}
        entry['filename'] = os.path.basename(file) #gets the filename without the dir
        entry['class'] = class_names[np.argmax(score)]
        entry['score'] = 100 * np.max(score)
        all_predictions.append(entry)
    df = pd.DataFrame(all_predictions)
    if out_dir is None:
                out_dir = os.path.dirname(os.path.realpath(jpg_dir))
    filename = f"{Path(jpg_dir).stem}_prediction_classifer.csv"
    csv_path = f"{out_dir}/{filename}"
    df.to_csv(csv_path)
    print(f"\nThe csv_file {filename} has been successfully saved in {out_dir}")
    return df


#--------------------------------Save Pictures--------------------------------#


def create_dirs(dataframe: pd.DataFrame, path: str) -> None:
    """
    Creates for every class a seperate directory.

    Args:
        dataframe (pd.Dataframe): dataframe containig the classes as a column
        path (str): path of chosen directory
    """
    uniques = dataframe["class"].unique()
    for uni_class in uniques:
        Path(f"{path}/{uni_class}").mkdir(parents=True, exist_ok=True)

#TODO why call this label_id?
def make_file_name(label_id: str, pic_class: str, occurence: int) -> str:
    """
    Creates a fitting filename.

    Args:
        label_id (str): string containing the label id.
        pic_class (str): class of the label.
        occurence (int): counts how many times the label class already
                         occured in the picture.
    """
    label_id = re.sub(r"_+label", "", label_id)
    filename = f"{label_id}_label_{pic_class}_{occurence}.jpg"
    return filename


def rename_picture(img_raw: np.ndarray , path: str,
                 filename: str,pic_class: str) -> None:
    """
    Renames the pictures using the given class and their occurence.

    Args:
        img_raw (numpy.matrix): input jpg converted to numpy matrix by cv2.
        path (str): path where the picture should be saved.
        filename (str): name of the picture.
        pic_class (str): class of the label.
    """
    filepath=f"{path}/{pic_class}/{filename}"
    cv2.imwrite(filepath, img_raw)


def filter_pictures(jpg_dir: Path, dataframe: str,
                 out_dir: Path = Path(os.getcwd())) -> None:
    """
    Creates new folders for each class of the newly named classified pictures.

    Args:
        jpg_dir (str): path to directory with jpgs.
        dataframe (str): path to csv file.
        out_dir (str): path to the target directory to save the cropped jpgs.
    """
    dir_path = jpg_dir
    out_dir = out_dir
    create_dirs(dataframe, out_dir) #creates dirs for every class

    for filepath in glob.glob(os.path.join(dir_path, '*.jpg')):
            filename = os.path.basename(filepath)
            match = dataframe[dataframe.filename == filename]
            image_raw = utils.load_jpg(filepath) # image_raw = utils.load_jpg(filepath)
            label_id = Path(filename).stem
            classes = []
            for _,row in match.iterrows():
                pic_class = row['class']
                occ = classes.count(pic_class) + 1
                filename = make_file_name(label_id, pic_class, occ)
                rename_picture(image_raw,out_dir,filename,pic_class)
    print(f"\nThe images have been successfully saved in {out_dir}")

