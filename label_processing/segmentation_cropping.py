# Import Librairies
import cv2
#import sys
import re
#import detecto
import torch
import os
import glob
import detecto.utils
import concurrent.futures
import pandas as pd
import numpy as np

from pathlib import Path
from detecto.core import Model
#from torchvision import transforms
import label_processing.utils


#---------------------Image Segmentation---------------------#

class PredictLabel():

    def __init__(self, path_to_model: str , classes: list,
                 jpg_path: str|Path|None = None,
                 threshold: float  = 0.8) -> None:
        """
        Init Method for the Pscrop.redict labels Class.

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
        self.jpg_path = jpg_path
        self.threshold = threshold
        self.model = self.retrieve_model()

        
    @property
    def jpg_path(self):
        return self._jpeg_path
    
    @jpg_path.setter
    def jpg_path(self, jpg_path: str|Path):
        if jpg_path == None:
            self._jpg_path = None
        elif (isinstance(isinstance, str)):
            self._jpg_path = Path(jpg_path)
        elif (isinstance(jpg_path,Path)):
            self._jpg_path = jpg_path
            
    def retrieve_model(self) -> None:
        """
        Call trained object detection model, for example *model_labels_class.pth*.
        The model was trained with the Detecto python package which is built on top
        of PyTorch.
            
        Returns:
            model (detecto.core.Model): trained object detection model.
        """
        model_type = Model.DEFAULT
        model = Model(self.classes, model_name=model_type)
        model.get_internal_model().load_state_dict(torch.load(
            self.path_to_model, map_location=model._device),
                                                strict=False
                                                )
        return model
    
    def class_prediction(self, jpg_path: Path = None) -> dict[str, str, str,
                                                     int, int, int, int]:
        """
        Uses the trained model created by Detecto and tries to predict the 
        labelling of all files in a directory. It then returns a Pandas Dataframe.

        Args:
            model(detecto.core.Model): access to object detection model and 
            pretrained PyTorch model (fasterrcnn_resnet50_fpn).
                                       
        Returns:
            DataFrame (pd.Dataframe): pandas Dataframe with the results.
        """
        if jpg_path is None:
            jpg_path = self.jpg_path
        image = detecto.utils.read_image(str(jpg_path))
        predictions = self.model.predict(image)
        labels, boxes, scores = predictions
        for i, labelname in enumerate(labels):
            entry = {}
            entry['filename'] = jpg_path.name
            entry['class'] = labelname
            entry['score'] = scores[i]
            entry['xmin'] = boxes[i][0]
            entry['ymin'] = boxes[i][1]
            entry['xmax'] = boxes[i][2]
            entry['ymax'] = boxes[i][3]
        return entry
    
def prediction_parallel(jpg_dir: Path, predictor: PredictLabel, n_processes: int):
    file_names: list[Path] = list(jpg_dir.glob("*.jpg"))

    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = executor.map(predictor.class_prediction, file_names)
    
    return pd.DataFrame(list(results))


def clean_predictions(jpg_dir: Path, dataframe: pd.DataFrame,
                        threshold: float, out_dir = None) -> pd.DataFrame:
    """
    Creates a clean dataframe only with boxes coordinates exceeding a 
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
    dataframe = dataframe.loc[dataframe['score'] >= threshold]
    dataframe[['xmin', 'ymin','xmax','ymax']] = \
        dataframe[['xmin', 'ymin','xmax','ymax']].fillna('0')
    if out_dir is None:
        parent_dir = jpg_dir.resolve().parent  #get parent of jpg_dir
    else:
        parent_dir = out_dir
    filename = f"{jpg_dir.stem}_predictions.csv"
    csv_path = f"{parent_dir}/{filename}"
    dataframe.to_csv(csv_path)
    print(f"\nThe csv_file {filename} has been successfully saved in {out_dir}")
    return dataframe


#---------------------Image Cropping---------------------#    
    
def crop_picture(img_raw: np.ndarray , path: str,
                 filename: str,pic_class: str,**coordinates) -> None:
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


def make_file_name(label_id: str, pic_class: str, occurence: int) -> None:
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

def create_dirs(dataframe: pd.DataFrame, path: str) -> None:
    """
    Creates for every class a seperate directory.
    In image preprocessing, erosion and dilation are often
    combined in the presented order to remove noise.
scrop.
    Args:
        dataframe (pd.Dataframe): dataframe containig the classes as a column
        path (str): path of chosen directory
    """
    uniques = dataframe["class"].unique()
    for uni_class in uniques:
        Path(f"{path}/{uni_class}").mkdir(parents=True, exist_ok=True)

def create_crops(jpg_dir: Path, dataframe: str,
                 out_dir: Path = Path(os.getcwd())) -> None:
    """
    Creates crops by using the csv from applying the model and the original
    pictures inside a directory.

    Args:
        jpg_dir (): path to directory with jpgs.
        dataframe (str): path to csv file.
        out_dir (Path): path to the target directory to save the cropped jpgs.
    """
    dir_path = jpg_dir
    new_dir_name = Path(dir_path.name + "_cropped")
    path = out_dir.joinpath(new_dir_name)
    path.mkdir(parents=True, exist_ok=True)
    create_dirs(dataframe, path) #creates dirs for every class
    for filepath in glob.glob(os.path.join(dir_path, '*.jpg')):
        filename = os.path.basename(filepath)
        match = dataframe[dataframe.filename == filename]
        image_raw = label_processing.utils.load_jpg(filepath)
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
    print(f"\nThe images have been successfully saved in {path}")
    
