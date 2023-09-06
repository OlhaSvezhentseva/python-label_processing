#Import Librairies
import json
import pandas as pd


def dump_json(data, file_name: str) -> str:
    data = json.dumps(data, indent=4)
    with open(file_name, "w") as outfile:
        outfile.write(data)
    return f"Data saved in {file_name}"


def load_json(file: str):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def read_vocabulary(file: str) -> dict:
    """The function reads a csv with vocabulary."""
    voc = pd.read_csv(file)
    return dict(voc.values)
