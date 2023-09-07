# Import Librairies
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


def clean_data(data):
    '''
    Dataset preprocessing

    Args:
        data (list of dict): List of dictionaries with labels' transcription

    Returns:
        list of dict: Preprocessed list of dictionaries
    '''
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


def redundancy(data):
    '''
    Calculate transcription redundancy in preprocessed dataset.

    Args:
        data (list of dict): Preprocessed list of dictionaries with labels' transcription

    Returns:
        list of dict: Preprocessed list of dictionaries with grouped duplicates
    '''
    data = clean_data(data)
    text_set = set()
    duplicates = []
    for item in data:
        text = item['text']
        if text in text_set:
            duplicates.append(item)
        text_set.add(text)
    return duplicates


def per_redundancy(data):
    '''
    Calculate percentage of transcription redundancy in preprocessed dataset with grouped duplicates.

    Args:
        data (list of dict): Preprocessed list of dictionaries with labels' transcription and grouped duplicates

    Returns:
        int: Percentage of redundant text
    '''
    data_clean = clean_data(data)
    duplicates = redundancy(data_clean)
    sum_text = len(data_clean)
    sum_dup = len(duplicates)  # Count duplicates
    percentage_red = round(sum_dup / sum_text * 100)
    return percentage_red

