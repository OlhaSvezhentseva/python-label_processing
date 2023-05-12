# Import Librairies
import os
import re
import json
import pandas as pd
import cv2
from typing import Optional

PATTERN = r"/u/|http|u/|coll|mfn|/u|URI"


#---------------------Check dir JPG---------------------#

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
    new_filename, _ = os.path.splitext(original_path)
    
    if original_path[-1] == "/" :
        new_filename = (f"{os.path.basename(os.path.dirname(new_filename))}"
                        f"_{appendix}")
    else:
        new_filename = f"{os.path.basename(original_path)}_{appendix}"
    
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

def get_nuri(data: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Correct NURIs' format in OCR transcription json file outputs "text.

    Args:
        data (list[dict[str,str]]): output of the ocr
        
    Returns:
        str: correct NURI formats in "text" in json file ocr output
    """
    new_data=data.copy()
    #search for NURI number in "ID"
    reg = re.compile(r"_u_[A-Za-z0-9]+") 
    for item, new_item in zip(data, new_data):
        findString = item["text"]
        findNURI = item["ID"]
        if check_text(findString): #checks if label is a NURI - True/False
            try:
                NURI = reg.search(findNURI).group()
                replaceString = "http://coll.mfn-berlin.de/u/"+ NURI[3:]
                #replace "text" with NURI patterns formatted "ID"
                new_item["text"] = replaceString 
            except AttributeError:
                    pass
    return new_data

def load_dataframe(filepath_csv):
    """
    Loads the csv file using Pandas.

    Args:
        filepath_csv(str): string containing the path to the csv with the
                           results from applying the model.
                           
    Returns:
        pd.Dataframe: The csv as a Pandas Dataframe
    """
    dataframe = pd.read_csv(filepath_csv)
    return dataframe


def load_jpg(filepath):
    """
    Loads the jpg file using the opencv module.

    Returns:
        Mat: cv2 image object
    """
    jpg = cv2.imread(filepath)
    return jpg
