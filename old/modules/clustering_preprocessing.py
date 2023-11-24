#Import Librairies
import pandas as pd
from json import loads, dumps
import warnings
warnings.filterwarnings('ignore')


def cluster_ID(json: str) -> pd.DataFrame:
    """
    Uses the json file OCR output to preprocess it before clustering.
    It adds a new key "label_ID", a unique identifier for the outputs coming from the same picture
    and group them together.

    Args:
        json (str): path to the OCR json file.
                                       
    Returns:
        df (pd.DataFrame): pandas Dataframe with cluster_ID new key.
    """
    df = pd.read_json(json)
    df["label_ID"] = df['ID']
    df['label_ID'] = df['label_ID'].str.replace('_typed_\d+\.jpg','')
    df['cluster_ID'] = pd.factorize(df.label_ID)[0]
    df['cluster_ID'] = 'label_' + df['cluster_ID'].astype(str)
    df = df.sort_values(by=['cluster_ID'])
    df.drop(columns=['label_ID'], inplace = True)
    return df

def df_to_json(json: str, cluster_json: str) -> str:
    """
    Save the pandas Dataframe has a new json file.

    Args:
        json (str): path to the OCR json file.
        cluster_json (str): path to where we want to save the preprocessed json file.   
                                
    Returns:
        json (str): preprocessed json file.
    """
    df = cluster_ID(json)
    result = df.to_json(orient='records')
    parsed = loads(result)
    json = dumps(parsed, indent=1, sort_keys=True)
    f = open(cluster_json,"x")
    f.write(json)
