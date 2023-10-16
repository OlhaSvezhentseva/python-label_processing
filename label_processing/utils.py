# Import Librairies
import os
import re
import json
import pandas as pd
import cv2
from typing import Optional
import numpy as np

PATTERN = r"/u/|http|u/|coll|mfn|/u|URI"


#---------------------Check dir JPEG---------------------#

def check_dir(dir) -> None:
    """
    Checks if the directory given as an argument contains jpg files.

    Args:
        dir (str): path to directory

    Raises:
        FileNotFoundError: raised if no jpg files are found in directory
    """
    if not any(file_name.endswith('.jpg') for file_name in os.listdir(dir)):
        raise FileNotFoundError(("The directory given does not contain "
                                 "any jpg-files. You might have chosen the wrong"
                                 "directory?")) 
        

#---------------------New Filename Preprocessed Images---------------------#


def generate_filename(original_path: str, appendix: str,
                      extension: Optional[str] = None) -> str:
    """
    Gets the path to a file or dictionary as an input and returns it with an 
    appendix added to the end.   

    Args:
        original_path (str): original path to file or directory 
        appendix (str): what needs to be appended
        extension (Optional[str]): either no extension (for directories) or a 
        file extension as a string 

    Returns:
        str: new file or directory name
    """
    #check if file is a dir and add apendix
    
    #remove extension if it has one
    new_filename, _ = os.path.splitext(os.path.basename(original_path))
    
    appendix = appendix.strip("_")
    if original_path[-1] == "/" :
        new_filename = (f"{os.path.basename(os.path.dirname(new_filename))}"
                        f"_{appendix}")
    else:
        new_filename = f"{new_filename}_{appendix}"
    
    if extension is not None:
        if extension[0] != ".":
            new_filename = f"{new_filename}.{extension}"
        else:
            new_filename = f"{new_filename}{extension}"
    return new_filename


#---------------------Save JSON---------------------#


def save_json(data: list[dict[str,str]], filename: str, path: str) -> None:
    """
    Saves a json file with human readable format.

    Args:
        data (list[dict[str,str]]): output of the ocr
        filename (str): how the json should be called
        path (str): path were the json should be saved
    """
    filepath = os.path.join(path, filename)
    with open(filepath, "w", encoding = 'utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4,
                      separators=(',', ': '))
        
        
#---------------------Check and correct NURIs---------------------#


def check_text(transcript: str) -> bool:
    """
    Check NURIs' format in OCR transcription json file outputs "text".

    Args:
        transcript: str: json file output of the ocr

    Returns:
        str: Boolean
    """
    #search for NURI patterns in "text"
    pattern = re.compile(PATTERN)
    match = pattern.search(transcript)
    return True if match else False

def replace_nuri(transcript: dict[str, str]) -> dict[str, str]:
    """
    Correct NURIs' format in OCR transcription json file outputs "text.

    Args:
        transcript (dict[str,str]): JSON transcript with "ID" and "text" fields.

    Returns:
        dict[str,str]: JSON transcript with corrected NURI formats in "text" field.
    """
    # Compile both patterns for NURI
    reg_nuri = re.compile(r"_u_[A-Za-z0-9]+")
    reg_picturae_nuri = re.compile(r"_u_([0-9a-fA-F]+)\.jpg")
    
    try:
        # Search for both NURI patterns in "ID"
        nuri = reg_nuri.search(transcript["ID"])
        picturae_nuri = reg_picturae_nuri.search(transcript["ID"])
        
        if nuri:
            # If the first pattern is found, replace it
            replace_string = "http://coll.mfn-berlin.de/u/" + nuri.group()[3:]
            transcript["text"] = replace_string
        elif picturae_nuri:
            # If the second pattern is found, replace it
            replace_string = "http://coll.mfn-berlin.de/u/" + picturae_nuri.group(1)
            transcript["text"] = replace_string
    except AttributeError:
        pass

    return transcript


def load_dataframe(filepath_csv: str) -> pd.DataFrame:
    """
    Loads the csv file using Pandas.

    Args:
        filepath_csv (str): string containing the path to the csv with the
                           results from applying the model.
                           
    Returns:
        pd.Dataframe: The csv as a Pandas Dataframe
    """
    dataframe = pd.read_csv(filepath_csv)
    return dataframe


def load_jpg(filepath: str) -> np.ndarray:
    """
    Loads the jpg files using the opencv module.

    Args:
        filepath (str): path to jpg files

    Returns:
        Mat (numpy.typing.NDArray): cv2 image object
    """
    jpg = cv2.imread(filepath)
    return jpg

def load_json(file: str):
    """
    Load JSON data from a file and deserialize it.

    Args:
        file (str): The name of the file containing JSON data.

    Returns:
        Any: The JSON data.
    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def read_vocabulary(file: str) -> dict:
    """
    Read a CSV file containing vocabulary and convert it to a dictionary.

    Args:
        file (str): The name of the CSV file containing vocabulary data.

    Returns:
        dict: A dictionary where keys and values are taken from the CSV data.
    """
    voc = pd.read_csv(file)
    return dict(voc.values)
