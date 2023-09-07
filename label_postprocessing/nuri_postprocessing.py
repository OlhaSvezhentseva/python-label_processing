# Import Librairies
import json
import ast
import os


def json_load(f: str) -> list:
    """
    Loads predictions from the OCR outputs as a json file.

    Args:
        f (str): path to the json file

    Returns:
        python_dict (list): list of dictionaries
    """

    with open(f, 'r') as f:
        data = json.load(f)
        data = json.dumps(data)
        python_dict = ast.literal_eval(data)
    return python_dict


def withoutNURIs(json: list) -> list:
    """
    Filters items in the list of dictionaries that are not starting with http.

    Args:
        f (list): list of dictionaries

    Returns:
        a (list): filtered list of dictionaries
    """
    data = json_load(json)
    result = []
    prefix = 'http'
    for item in data:
        if item['text'].startswith(prefix) is False:
            result.append(item)
    a = result
    return a


def withNURIs(json: str) -> list:
    """
    Filters items in the list of dictionaries that are starting with http.

    Args:
        f (list): list of dictionaries

    Returns:
        b (list): filtered list of dictionaries
    """
    data = json_load(json)
    result = []
    prefix = 'http'
    for item in data:
        if item['text'].startswith(prefix) is True:
            result.append(item)
    b = result
    return b


def dumps_json_with(f: list) -> str:
    """
    Saves filtered list of dictionaries with NURIS.

    Args:
        f (list): list of dictionaries with NURIS

    Returns:
        json_withNURIs (str): filtered json file
    """
    a = withNURIs(f)
    json_withNURIs = json.dumps(a)
    return json_withNURIs
    

def dumps_json_without(f: list) -> str:
    """
    Saves filtered list of dictionaries without NURIS.

    Args:
        f (list): list of dictionaries without NURIS

    Returns:
        json_withoutNURIs (str): filtered json file
    """
    a = withoutNURIs(f)
    json_withoutNURIs = json.dumps(a)
    return json_withoutNURIs


def write_json_with(f: str, filepath: str, filename="withNURIs.json") -> str:
    """
    Writes json file with NURIS.

    Args:
        f (str): filtered json file with NURIs
        filepath (str): path to saving directory

    Returns:
        f (str): json file
    """
    json = dumps_json_with(f)
    desired_dir = filepath
    full_path = os.path.join(desired_dir, filename)
    with open(full_path, 'w') as f:
        f.write(json)


def write_json_without(f: str, filepath: str, filename="withoutNURIs.json") -> str:
    """
    Writes json file without NURIS.

    Args:
        f (str): filtered json file without NURIs
        filepath (str): path to saving directory

    Returns:
        f (str): json file
    """
    json = dumps_json_without(f)
    desired_dir = filepath
    full_path = os.path.join(desired_dir, filename)
    with open(full_path, 'w') as f:
        f.write(json)
