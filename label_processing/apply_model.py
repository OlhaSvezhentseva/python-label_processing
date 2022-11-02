"""
Module containing all functions concerning the application of the 
model and to use the predicted coordinates for cropping the predicted labels.  
"""
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


class Predict_Labels():

    def __init__(self, path_to_model, classes, jpg_dir, threshold = 0.8):
        """
        Init Method for the Predict labels Class

        Args:
            path_to_model (str): string that contains the path to the model
            classes (list): list that contains the classes that should be used
            jpg_dir (str): string with path to the directory containing the 
                original jpgs
            threshold (float, optional): Threshold value for scores. 
                Defaults to 0.8.
        """
        self.path_to_model = path_to_model
        self.classes = classes
        self.jpg_dir = jpg_dir
        self.threshold = threshold
        
    def get_model(self):
        """
        Call trained object detection model *model_labels_class.pth*. 
        The model was trained with the Detecto python package which is built on top 
        of PyTorch.
            
        Returns:
            model : trained object detection model
            
        """
        print("Calling trained object detection model")
        model_type = Model.DEFAULT
        model = Model(self.classes, model_name=model_type)
        model.get_internal_model().load_state_dict(torch.load(
            self.path_to_model, map_location=model._device),
                                                strict=False
                                                )
        return model

    def class_prediction(self, model):
        """
        Uses the trained model created by Detecto and tries to predict boxes of all 
        files in a directory. It then returns a pandas Dataframe.

        Args:
            model(detecto.core.Model): access to object detection model 
                *model_labels_class.pth* and pretrained PyTorch model 
                (fasterrcnn_resnet50_fpn)

        Returns:
            DataFrame: pandas Dataframe with the results

        """
        all_predictions = []
        print("Predicting coordinates")
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

    def clean_predictions(self, dataframe):
        """
        Creates a clean dataframe only with boxes exceeding a given threshold score.

        Args:
            dataframe(pandas.DataFrame): pandas Dataframe with prediction 
                coordinates and scores labels

        Returns:
            DataFrame: pandas Dataframe with the trimmed results
        """
        print("Filter coordinates")
        dataframe = dataframe
        colnames = ['score','xmin', 'ymin', 'xmax', 'ymax']
        for header in colnames:
            dataframe[header] = dataframe[header].astype('str').str.\
                extractall('(\d+.\d+)').unstack().fillna('0').sum(axis=1).astype(float)
        dataframe = dataframe.loc[ dataframe['score'] >= self.threshold ]
        #TODO add "0" to empty cells
        #dataframe.fillna('0', inplace=True)
        location = os.path.dirname(os.path.realpath(self.jpg_dir)) 
        filename = f"{Path(self.jpg_dir).stem}_predictions.csv"
        csv_path = f"{location}/{filename}"
        dataframe.to_csv(csv_path)
        print(f"{filename} has been successfully saved in {location}")
        #returns csv_path as
        return dataframe

def load_dataframe(filepath_csv):
    """
    loads the csv file using pandas

    Args:
        filepath_csv(str): string containing the path to the csv with the 
            qresults from applying the model
        
    
    Returns:
        Dataframe: The csv as a pandas Dataframe-object
    """
    #NOTE: maybe this function is not necessary
    dataframe = pd.read_csv(filepath_csv)
    return dataframe


def load_jpgs(filepath):
    """
    Loads the jpg file using the opencv module

    Returns:
        dict: dictionary with filenames as keys and cv2.imread() outputs 
        as values 
    """
    with open(filepath) as f:
        jpg = cv2.imread(filepath)

    return jpg
    
    
def crop_picture(img_raw,path,filename,**coordinates):
    """crops the picture using the given coordinates

    Args:
        img_raw (numpy matrix): input jpg converted to numpy matrix by cv2
        path (str): path where the picture should be saved
        filename (str): name of the picture
    """
    xmin = coordinates['xmin']
    ymin = coordinates['ymin']
    xmax = coordinates['xmax']
    ymax = coordinates['ymax']
    filepath= path + filename
    crop = img_raw[ymin:ymax, xmin:xmax]
    cv2.imwrite(filepath, crop)



def make_file_name(label_id, pic_class, occurence):
    """
    Creates a fitting filename

    Args:
        label_id (str): string containing the label id 
        pic_class (str): class of the label
        occurence (int): count of how many times the label class already 
            occured in the picure
    """
    #remove occurences of _label in filename to make the name look nicer
    label_id = re.sub(r"_+label", "", label_id) 
    filename = f"{label_id}_label_{pic_class}_{occurence}.jpg"
    return filename

def create_crops(jpg_dir, dataframe):
    """
    creates crops by using the csv from applying the model and the original 
    pictures inside a directory

    Args:
        file (str): path to csv file 
        directory (str): path to directory with jpgs
    """
    #create a new_directory
    dir_path = jpg_dir
    if dir_path[-1] == "/" : #check if the directory parsed has a slash at the end
        new_dir = f"{os.path.basename(os.path.dirname(dir_path))}_cropped"
    else:
        new_dir = f"{os.path.basename(dir_path)}_cropped"
    cwd = os.getcwd()
    path = (f"{cwd}/{new_dir}/")
    #create directory
    Path(path).mkdir(parents=True, exist_ok=True)
    
    for filepath in glob.glob(os.path.join(dir_path, '*.jpg')):
        filename = os.path.basename(filepath)
        match = dataframe[dataframe.filename == filename]
        image_raw = load_jpgs(filepath)
        #sets the name for the dircetory, that should be created
        label_id = Path(filename).stem
        #list of classes in order to check for doubles 
        classes = []
        #loops through dataframe and returns every object as a Series
        for _,row in match.iterrows(): 
            pic_class = row['class']
            occ = classes.count(pic_class) + 1 
            #create filename
            filename = make_file_name(label_id, pic_class, occ)
            #coordinates as dic to parse them to crop_picture method
            coordinates = {'xmin':int(row.xmin),'ymin':int(row.ymin),
                           'xmax':int(row.xmax),'ymax':int(row.ymax)}
            crop_picture(image_raw,path,filename,**coordinates)
            classes.append(pic_class)
    print(f"The images have been successfully saved in {cwd}/{new_dir}")