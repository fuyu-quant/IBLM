import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple linear regression model for prediction
        # We are assuming that the target is a linear combination of the features a, b, c, d
        # The coefficients of the linear combination are determined by observing the data
        # We are assuming that the target is 1 if the linear combination is greater than 0.5 and 0 otherwise
        y = 0.2*row['a'] + 0.2*row['b'] + 0.2*row['c'] + 0.2*row['d']
        if y > 0.5:
            y = 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)