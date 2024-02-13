# manipulate data
import numpy as np
import pandas as pd

# visualize data
import matplotlib.pyplot as plt
import seaborn as sn

# model preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean(dataset=None:str) -> pd.DataFrame:
    match dataset:
        
        case "cvd":
            pass

        case "nf":
            pass

    
    return df_cleaned

def split(cleaned=None):

    return train, test

def scale(split_df=None, train=None:pd.DataFrame, test=None:pd.Series):

    match split_df:
        
        case 'cvd':
            pass

        case 'nf':
            pass
        
    return df_scaled

def baseline(y_train=None:pd.Series) -> float:

    return baseline_acc
    
def final_dataset(scaled=None:str) -> pd.DataFrame:

    match scaled:
        
        case 'cvd':
            pass

        case 'nf':
            pass

        return df_final
    
