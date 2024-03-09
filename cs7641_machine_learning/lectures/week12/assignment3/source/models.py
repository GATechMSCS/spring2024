# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# clustering
import clustering

# dimentionality reduction
import dimensionality_reduction

# non-linear
import non_linear

def put_it_all_together(df:pd.DataFrame()):
    
    steps = range(1, 6, 1)

    for step in steps:
            
        match step:
            case 1:
                print(step)

            case 2:
                print(step)

            case 3:
                print(step)

            case 4:
                print(step)

            case 5:
                print(step)

def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')
    
    put_it_all_together(df=X_train_scaled_cd)

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    put_it_all_together(df=X_train_scaled_nf)

if __name__ == "__main__":
    main()