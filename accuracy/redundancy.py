# Import librairies
import pandas as pd
import re

PATH = "/Users/Margot/Desktop/ground_truth.csv"

df = pd.read_csv(PATH, sep= ";")

def clean_data(df):
	df['text'] = df['text'].str.lower() #remove lowercase
	df['text'] = df['text'].str.replace('[^\w\s]','') #remove punctuation
	df['text'] = df['text'].str.replace(' ', '') #remove whitespace
	df['text'] = df['text'].str.replace(r'httpcollmfnberlindeu([0-9a-zA-Z]+)', 'httpcollmfnberlindeu') #remove NURIs numbers
	df['text'] = df['text'].str.replace('\d+', '') #remove digits
	return df

def redundancy(df):
	df = clean_data(df)
	duplicates = df["text"]
	df[duplicates.isin(duplicates[duplicates.duplicated()])].sort_values("text") #groupby duplicates
	df = pd.concat(g for _, g in df.groupby("text") if len(g) > 1)
	return df

def sum_redundancy(df):
    df = redundancy(df)
    result = df["text"].duplicated().sum() #find sum of duplicates
    return print("Redundant text:", result)

    