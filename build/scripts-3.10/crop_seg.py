#!python

'''
Execute the segmentation_cropping.py module.
'''

#Import module from this package
import label_processing.segmentation_cropping as scrop
import label_processing
#import third party libraries
import argparse
import os
import warnings
import glob
import pandas as pd
warnings.filterwarnings('ignore')

from pathlib import Path
import concurrent.futures


def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'crop_seg.py [-h] [-c N] -m <model/number> -j </path/to/jpgs> -o </path/to/jpgs_outputs> '
    parser =  argparse.ArgumentParser(description=__doc__,
            add_help = False,
            usage = usage
            )

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )
    parser.add_argument(
            '-o', '--out_dir',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the resulting crops and the csv will be stored.\n'
                  'Default is the user current working directory.')
            )
    
    parser.add_argument(
            '-j', '--jpg_dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the jpgs are stored.')
            )
    
    parser.add_argument( 
            '-m', '--model',
            metavar='',
            choices = range(1,4),
            type=int,
            default = 1,
            action='store',
            help=('Optional argument: select which segmenmtation-model should be used.\n'
                 '1 : only box.\n'
                 '2 : classes (location, nuri, uce, taxonomy, antweb, casent_number, dna, other, collection).\n'
                 '3 : handwritten/typed.\n'
                 'Default is only box')
            )
    
    args = parser.parse_args()

    return args

def get_modelnum(model_int: int) -> str:
    """
    Returns the chosen model.

    Args:
         model_int (int): integer for model selection.

    Returns:
         path (str): path to the selected model.
    """
    script_dir = os.path.dirname(__file__)
    rel_path1 = "../models/model_labels_box.pth"
    model_file1 = os.path.join(script_dir, rel_path1)
    rel_path2 = "../models/model_labels_class.pth"
    model_file2 = os.path.join(script_dir, rel_path2)
    rel_path3 = "../models/model_labels_h_t.pth"
    model_file3 = os.path.join(script_dir, rel_path3)
    if model_int == 1:
        return model_file1
    elif model_int == 2:
        return model_file2
    else:
        return model_file3

def get_classtype(model_int: int) -> list:
    """
    Returns the chosen classes.

    Args:
        class_int (int): integer for class selection.

    Returns:
        list: list with the selected classes.
    """
    if model_int == 1:
        return ["box"]
    elif model_int == 2:
        return ["location", "nuri", "uce", "taxonomy", "antweb",
                "casent_number", "dna", "other", "collection"]
    else:
        return ["handwritten", "typed"]


def create_crops(jpg_dir: str, dataframe: str,
                 out_dir: str = os.getcwd()) -> None:
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
    scrop.create_dirs(dataframe, path) #creates dirs for every class
    for filepath in glob.glob(os.path.join(dir_path, '*.jpg')):
        filename = os.path.basename(filepath)
        match = dataframe[dataframe.filename == filename]
        image_raw = label_processing.utils.load_jpg(filepath)
        label_id = Path(filename).stem
        classes = []
        for _,row in match.iterrows(): 
            pic_class = row['class']
            occ = classes.count(pic_class) + 1 
            filename = scrop.make_file_name(label_id, pic_class, occ)
            coordinates = {'xmin':int(row.xmin),'ymin':int(row.ymin),
                           'xmax':int(row.xmax),'ymax':int(row.ymax)}
            scrop.crop_picture(image_raw,path,filename,pic_class,**coordinates)
            classes.append(pic_class)
    print(f"\nThe images have been successfully saved in \
        {os.path.join(out_dir, new_dir)}")
        

    

# does not execute main if the script is imported as a module
if __name__ == '__main__': 
    args = parsing_args()
    model_path = get_modelnum(args.model)
    jpeg_dir = args.jpg_dir
    classes = get_classtype(args.model)
    out_dir = args.out_dir
    
    predictor = scrop.PredictLabels(model_path, classes, jpeg_dir)
    
    
    # 2. Model Predictions
    df = scrop.prediction_parallel(args.jpg_dir,predictor)
    
    # 3. Filter model predictions and save csv
    df = scrop.clean_predictions(df, out_dir = out_dir)
    
    # 4. Cropping
    scrop.create_crops(jpeg_dir, df, out_dir = out_dir)
    

