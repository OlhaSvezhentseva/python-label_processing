#Import Librairies
import json
import pandas as pd


def dump_json(data, file_name: str) -> str:
    """
    Serialize Python data to JSON and save it to a file.

    Args:
        data: Any valid JSON Python data.
        file_name (str): The name of the file to save the JSON data.

    Returns:
        str: A message indicating the success of saving the data.
    """
    data = json.dumps(data, indent=4)
    with open(file_name, "w") as outfile:
        outfile.write(data)
    return f"Data saved in {file_name}"

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
