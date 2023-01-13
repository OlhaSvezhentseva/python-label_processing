#!python
'''
Execute the segmentation_cropping.py, preprocessing_ocr.py and ocr_pytesseract.py scripts.
Takes as inputs: the path to the jpgs (jpg_dir),
                 the path to the model (model),
                 the classes used for the model (classes),
                 the path to the directory in which the resulting crops and the csv will be stored (out_dir),
                 the path to the directory in which the OCR outputs will be stored (out_dir_OCR) and
                 the path to the directory in which the OCR outputs of the preprocessed images will be stored (out_dir_OCR_pre)

Outputs: - the labels in the pictures are segmented and cropped out of the picture,
           becoming their own file named after their jpg of origin and class.
         - the segmentation outputs are also saved as a dataframe (filename, class, prediction score, coordinates).
         - the OCR outputs after performing on the cropped images.
         - the preprocessed images in their own _pre folder in the main input images directory.
         - the OCR outputs after performing on the preprocessed cropped images.
'''

#Import Librairies
import segmentation_cropping
import ocr_pytesseract
import preprocessing_ocr

import argparse
from pathlib import Path
import os
import urllib3
urllib3.disable_warnings()
import warnings
warnings.filterwarnings('ignore')

CLS_AMOUNT = 3

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'OCR2data.py [-h] -c 3 -j /path/to/jpgs -m /path/to/model -o /path/to/jpgs_outputs -o_OCR /path/to/OCR_output_file/outputOCR.txt o_OCR_pre /path/to/OCR_output_preprocessed_file/outputOCRpre.txt'
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
            '-c', '--classes',
            metavar='',
            choices = range(1,4),
            type=int,
            default = 1,
            action='store',
            help=('Optional argument: select which classes should be used according to the model selected.'
                 '1 : only box.'
                 '2 : classes (location, nuri, uce, taxonomy, antweb, casent_number, dna, other, collection).'
                 '3 : handwritten/typed.'
                 'Default is only box')
            )
    
    parser.add_argument(
            '-j', '--jpg_dir',
            metavar='',
            type=str,
            required = True,
            help='Directory where the jpgs are stored.'
            )
    
    parser.add_argument(
            '-m', '--model',
            metavar='',
            type=str,
            required = True,
            help='Path to the model to be used.'
            )
    
    parser.add_argument(
            '-o', '--out_dir',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the resulting crops and the csv will be stored.'
                  'Default is the user current working directory.')
            )
    
    parser.add_argument(
            '-o_OCR', '--out_dir_OCR',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the OCR outputs will be saved.'
                  'Default is the user current working directory.'))
    
    parser.add_argument(
            '-o_OCR_pre', '--out_dir_OCR_pre',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the OCR outputs of the preprocessed images will be saved.'
                  'Default is the user current working directory.')
                  
    )

    
    args = parser.parse_args()

    return args


def get_classtype(class_int):
    """
    Returns the chosen classes.

    Args:
        class_int (int): integer for class selection.

    Returns:
        list: list with the selected classes.
    """
    if class_int == 1:
        return ["box"]
    elif class_int == 2:
        return ["location", "nuri", "uce", "taxonomy", "antweb",
                "casent_number", "dna", "other", "collection"]
    else:
        return ["handwritten", "typed"]

# does not execute main if the script is imported as a module
if __name__ == '__main__': 
    args = parsing_args()
    model_path = args.model
    jpeg_dir = args.jpg_dir
    classes = get_classtype(args.classes)
    
    predictions = segmentation_cropping.Predict_Labels(model_path, classes, jpeg_dir)
    
    # 1. Call Model
    model = predictions.get_model()
    
    # 2. Model Predictions
    df = predictions.class_prediction(model)
    
    # 3. Filter model predictions and save csv
    df = predictions.clean_predictions(df, out_dir = args.out_dir)
    
    # 4. Cropping
    segmentation_cropping.create_crops(jpeg_dir, df, out_dir = args.out_dir)
    
    # 5. OCR - without image preprocessing
    dir_path = jpeg_dir
    out_dir = args.out_dir
    if dir_path[-1] == "/" :
        new_dir = f"{os.path.basename(os.path.dirname(dir_path))}_cropped"
    else:
        new_dir = f"{os.path.basename(dir_path)}_cropped"
    path = (f"{out_dir}/{new_dir}/")
    ocr_pytesseract.OCR(new_dir, path, out_dir_OCR = args.out_dir_OCR)
    
    # 6. OCR - with image preprocessing
    if dir_path[-1] == "/" :
        new_dir_pre = f"{os.path.basename(os.path.dirname(dir_path))}_pre"
    else:
        new_dir_pre = f"{os.path.basename(dir_path)}_pre"
    pre_path = (f"{out_dir}/{new_dir_pre}/")
    Path(pre_path).mkdir(parents=True, exist_ok=True)
    preprocessing_ocr.Preprocessing(path, pre_path)
    preprocessing_ocr.OCR(pre_path, out_dir_OCR_pre = args.out_dir_OCR_pre)

