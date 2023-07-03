import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Simple logic: if the sum of a, b, c, d is positive, predict high probability for target 1
        # if the sum is negative, predict low probability for target 1
        sum_row = row['a'] + row['b'] + row['c'] + row['d']
        if sum_row > 0:
            y = 0.9  # high probability for target 1
        else:
            y = 0.1  # low probability for target 1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)