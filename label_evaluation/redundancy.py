# Import Librairies
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Dataset preprocessing

    Args:
        DataFrame(pd.DataFrame): Pandas Dataframe with labels' transcription

    Returns:
        DataFrame (pd.DataFrame): Preprocessed Pandas Dataframe
    '''
    df['text'] = df['text'].str.lower() #remove lowercase
    df['text'] = df['text'].str.replace('[^\w\s]','') #remove punctuation
    df['text'] = df['text'].str.replace(' ', '') #remove whitespace
    patternDel = "http"
    filter = df['text'].str.contains(patternDel)
    df = df[~filter] #remove NURIs
    return df

def redundancy(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate transcription redundancy in preprocessed dataset.

    Args:
        DataFrame(pd.DataFrame): Preprocessed Pandas Dataframe with labels' transcription

    Returns:
        DataFrame (pd.DataFrame): Preprocessed Pandas Dataframe with grouped duplicates
    '''
    df = clean_data(df)
    duplicates = df["text"]
    df[duplicates.isin(duplicates[duplicates.duplicated()])].sort_values("text") #groupby duplicates
    df = pd.concat(g for _, g in df.groupby("text") if len(g) > 1)
    return df

def per_redundancy(df: pd.DataFrame) -> int:
    '''
    Calculate percentage of transcription redundancy in preprocessed dataset with grouped duplicates.

    Args:
        DataFrame(pd.DataFrame): Preprocessed Pandas Dataframe with labels' transcription and grouped duplicates

    Returns:
        String (int): Percentage redundant text
    '''
    df = pd.read_csv(df, sep= ";")
    df_clean = df
    df = redundancy(df)
    sum_text = df_clean["text"].value_counts().sum()
    sum_dup = df["text"].duplicated().sum() #find sum of duplicates
    percentage_red = round(sum_dup/sum_text*100)
    return percentage_red

