# Import librairies
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


def clean_data(df):
    df['text'] = df['text'].str.lower() #remove lowercase
    df['text'] = df['text'].str.replace('[^\w\s]','') #remove punctuation
    df['text'] = df['text'].str.replace(' ', '') #remove whitespace
    patternDel = "http"
    filter = df['text'].str.contains(patternDel)
    df = df[~filter] #remove NURIs
    #df['text'] = df['text'].str.replace(r'httpcollmfnberlindeu([0-9a-zA-Z]+)', 'httpcollmfnberlindeu') #remove NURIs numbers - completly?
    #df['text'] = df['text'].str.replace('\d+', '') #remove digits? y/n
    return df

def redundancy(df):
    df = clean_data(df)
    duplicates = df["text"]
    df[duplicates.isin(duplicates[duplicates.duplicated()])].sort_values("text") #groupby duplicates
    df = pd.concat(g for _, g in df.groupby("text") if len(g) > 1)
    return df

def per_redundancy(df):
    df = pd.read_csv(df, sep= ";")
    df_clean = df
    df = redundancy(df)
    sum_text = df_clean["text"].value_counts().sum()
    sum_dup = df["text"].duplicated().sum() #find sum of duplicates
    percentage_red = round(sum_dup/sum_text*100)
    return print("Percentage redundant text:", "%s%%"%percentage_red)

    
