# Import third-party libraries
import pandas as pd
import re
import warnings

# Suppress warning messages during execution
warnings.filterwarnings('ignore')


def clean_data(data: list[dict]) -> list[dict]:
    """
    Preprocess the dataset by converting text to lowercase, removing punctuation and whitespace,
    and excluding entries containing 'http'.

    Args:
        data (list of dict): List of dictionaries with labels' transcription.

    Returns:
        list of dict: Preprocessed list of dictionaries.
    """
    cleaned_data = []  # Create a new list to store cleaned data
    for item in data:
        text = item['text']
        cleaned_text = text.lower()  # Convert text to lowercase
        cleaned_text = ''.join(e for e in cleaned_text if e.isalnum() or e.isspace())  # Remove punctuation
        cleaned_text = cleaned_text.replace(' ', '')  # Remove whitespace
        if 'http' not in cleaned_text:
            item['text'] = cleaned_text  # Update the text field in the original dictionary
            cleaned_data.append(item)  # Add the cleaned dictionary to the new list
    return cleaned_data


def redundancy(data: list[dict]) -> list[dict]:
    """
    Calculate transcription redundancy in a preprocessed dataset by identifying duplicate entries.

    Args:
        data (list of dict): Preprocessed list of dictionaries with labels' transcription.

    Returns:
        list of dict: Preprocessed list of dictionaries with grouped duplicates.
    """
    data = clean_data(data)
    text_set = set()
    duplicates = []
    for item in data:
        text = item['text']
        if text in text_set:
            duplicates.append(item)
        text_set.add(text)
    return duplicates


def per_redundancy(data: list[dict]) -> int:
    """
    Calculate the percentage of transcription redundancy in a preprocessed dataset with grouped duplicates.

    Args:
        data (list of dict): Preprocessed list of dictionaries with labels' transcription and grouped duplicates.

    Returns:
        int: Percentage of redundant text.
    """
    data_clean = clean_data(data)
    duplicates = redundancy(data_clean)
    sum_text = len(data_clean)
    sum_dup = len(duplicates)  # Count duplicates
    percentage_red = round(sum_dup / sum_text * 100)
    return percentage_red

