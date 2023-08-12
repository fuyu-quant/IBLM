import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Based on the given data, it seems that when Feature_1 is greater than 0.5 and Feature_2 is less than 0.5, the target is more likely to be 1.
        # Conversely, when Feature_1 is less than 0.5 and Feature_2 is greater than 0.5, the target is more likely to be 0.
        # We can use these observations to predict the probability of the target being 1.
        
        if row['Feature_1'] > 0.5 and row['Feature_2'] < 0.5:
            y = 0.75
        elif row['Feature_1'] < 0.5 and row['Feature_2'] > 0.5:
            y = 0.25
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)