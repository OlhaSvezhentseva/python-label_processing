#Import Librairies
import pandas as pd
from json import loads, dumps
import warnings
warnings.filterwarnings('ignore')


def cluster_ID(json):
    df = pd.read_json(json)
    df["label_ID"] = df['ID']
    df['label_ID'] = df['label_ID'].str.replace('_typed_\d+\.jpg','')
    df['cluster_ID'] = pd.factorize(df.label_ID)[0]
    df['cluster_ID'] = 'label_' + df['cluster_ID'].astype(str)
    df = df.sort_values(by=['cluster_ID'])
    df.drop(columns=['label_ID'], inplace = True)
    return df

def df_to_json(json, cluster_json):
    df = cluster_ID(json)
    result = df.to_json(orient='records')
    parsed = loads(result)
    json = dumps(parsed, indent=1, sort_keys=True)
    f = open(cluster_json,"x")
    f.write(json)
