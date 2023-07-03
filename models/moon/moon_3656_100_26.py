import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple logic to predict the target.
        # If Feature_1 is greater than 1 and Feature_2 is less than 0, we predict the target as 1 (high probability).
        # Otherwise, we predict the target as 0 (low probability).
        if row['Feature_1'] > 1 and row['Feature_2'] < 0:
            y = 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)