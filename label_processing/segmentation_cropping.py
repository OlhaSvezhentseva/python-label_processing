"""
Module containing all functions concerning the application of the segmenation
models and the use of the predicted coordinates for cropping the labels.  
"""

# Import Librairies
import glob
import cv2
import os
#import sys
import re
#import detecto
import torch

import pandas as pd

from pathlib import Path
from detecto import utils
from detecto.core import Model
#from torchvision import transforms


#---------------------Image Segmentation---------------------#

class Predict_Labels():

    def __init__(self, path_to_model: str , classes: list, jpg_dir: str, threshold = 0.8: float):
        """
        Init Method for the Predict labels Class.

        Args:
            path_to_model (str): string that contains the path to the model.
            classes (list): list that contains the classes that should be used.
            jpg_dir (str): string with path to the directory containing the
                           original jpgs.
            threshold (float, optional): threshold value for scores.
                                         Defaults to 0.8.
        """
        self.path_to_model = path_to_model
        self.classes = classes
        self.jpg_dir = jpg_dir
        self.threshold = threshold
        
    def get_model(self):
        """
        Call trained object detection model, for example *model_labels_class.pth*.
        The model was trained with the Detecto python package which is built on top
        of PyTorch.
            
        Returns:
            model: trained object detection model.
        """
        print("\nCalling trained object detection model")
        model_type = Model.DEFAULT
        model = Model(self.classes, model_name=model_type)
        model.get_internal_model().load_state_dict(torch.load(
            self.path_to_model, map_location=model._device),
                                                strict=False
                                                )
        return model

    def class_prediction(self, model):
        """
        Uses the trained model created by Detecto and tries to predict the 
        labelling of all files in a directory. It then returns a Pandas Dataframe.

        Args:
            model(detecto.core.Model): access to object detection model and 
            pretrained PyTorch model (fasterrcnn_resnet50_fpn).
                                       
        Returns:
            DataFrame: pandas Dataframe with the results.
        """
        all_predictions = []
        print("\nPredicting coordinates")
        for file in glob.glob(f"{self.jpg_dir}/*.jpg"):
            image = utils.read_image(file)
            predictions = model.predict(image)
            labels, boxes, scores = predictions
            for i, labelname in enumerate(labels):
                entry = {}
                entry['filename'] = os.path.basename(file)
                entry['class'] = labelname
                entry['score'] = scores[i]
                entry['xmin'] = boxes[i][0]
                entry['ymin'] = boxes[i][1]
                entry['xmax'] = boxes[i][2]
                entry['ymax'] = boxes[i][3]
                all_predictions.append(entry)
        dataframe = pd.DataFrame(all_predictions)
        return dataframe

    def clean_predictions(self, dataframe: pd.DataFrame, out_dir = None) -> pd.DataFrame:
        """
        Creates a clean dataframe only with boxes´ coordinates exceeding a 
        given threshold score.

        Args:
            DataFrame (pd.DataFrame): Pandas Dataframe with predicted 
            coordinates and labels' scores.
            
        Returns:
            DataFrame (pd.DataFrame): Pandas Dataframe with the trimmed results.
        """
        print("\nFilter coordinates")
        colnames = ['score', 'xmin', 'ymin', 'xmax', 'ymax']
        for header in colnames:
            dataframe[header] = dataframe[header].astype('str').str.\
                extractall('(\d+.\d+)').unstack().fillna('').sum(axis=1).astype(float)
        dataframe = dataframe.loc[dataframe['score'] >= self.threshold]
        dataframe[['xmin', 'ymin','xmax','ymax']] = \
            dataframe[['xmin', 'ymin','xmax','ymax']].fillna('0')
        if out_dir is None:
            out_dir = os.path.dirname(os.path.realpath(self.jpg_dir))
        filename = f"{Path(self.jpg_dir).stem}_predictions.csv"
        csv_path = f"{out_dir}/{filename}"
        dataframe.to_csv(csv_path)
        print(f"\nThe csv_file {filename} has been successfully saved in {out_dir}")
        return dataframe


#---------------------Image Cropping---------------------#    
    
def crop_picture(img_raw: numpy.matrix, path: str, filename: str,pic_class: str,**coordinates):
    """
    Crops the picture using the given coordinates.

    Args:
        img_raw (numpy.matrix): input jpg converted to numpy matrix by cv2.
        path (str): path where the picture should be saved.
        filename (str): name of the picture.
        pic_class (str): class of the label.
    """
    xmin = coordinates['xmin']
    ymin = coordinates['ymin']
    xmax = coordinates['xmax']
    ymax = coordinates['ymax']
    filepath=f"{path}/{pic_class}/{filename}"
    crop = img_raw[ymin:ymax, xmin:xmax]
    cv2.imwrite(filepath, crop)


def make_file_name(label_id: str, pic_class: str, occurence: int):
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

def create_dirs(dataframe: pd.Dataframe, path: str):
    """
    Creates for every class a seperate directory.
    In image preprocessing, erosion and dilation are often
    combined in the presented order to remove noise.
    Args:
        dataframe (pd.Dataframe): dataframe containig the classes as a column
        path (str): path of chosen directory
    """
    uniques = dataframe["class"].unique()
    for uni_class in uniques:
        Path(f"{path}/{uni_class}").mkdir(parents=True, exist_ok=True)
    

def create_crops(jpg_dir: str, dataframe: str, out_dir = os.getcwd(): str):
    """
    Creates crops by using the csv from applying the model and the original
    pictures inside a directory.

    Args:
        jpg_dir (str): path to directory with jpgs.
        dataframe (str): path to csv file.
        out_dir (str): path to the target directory to save the cropped jpgs.
    """
    dir_path = jpg_dir
    if dir_path[-1] == "/" :
        new_dir = f"{os.path.basename(os.path.dirname(dir_path))}_cropped"
    else:
        new_dir = f"{os.path.basename(dir_path)}_cropped"
    path = (f"{out_dir}/{new_dir}/")
    Path(path).mkdir(parents=True, exist_ok=True)
    create_dirs(dataframe, path) #creates dirs for every class
    for filepath in glob.glob(os.path.join(dir_path, '*.jpg')):
        filename = os.path.basename(filepath)
        match = dataframe[dataframe.filename == filename]
        image_raw = utils.load_jpg(filepath)
        label_id = Path(filename).stem
        classes = []
        for _,row in match.iterrows(): 
            pic_class = row['class']
            occ = classes.count(pic_class) + 1 
            filename = make_file_name(label_id, pic_class, occ)
            coordinates = {'xmin':int(row.xmin),'ymin':int(row.ymin),
                           'xmax':int(row.xmax),'ymax':int(row.ymax)}
            crop_picture(image_raw,path,filename,pic_class,**coordinates)
            classes.append(pic_class)
    print(f"\nThe images have been successfully saved in \
        {os.path.join(out_dir, new_dir)}")
