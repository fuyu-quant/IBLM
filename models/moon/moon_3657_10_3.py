import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple linear regression model for prediction
        # y = mx + c, where m is the slope and c is the intercept
        # We are assuming m = 0.5 and c = 0.5 for simplicity
        m = 0.5
        c = 0.5
        y = m * row['Feature_1'] + m * row['Feature_2'] + c
        y = 1 / (1 + np.exp(-y))  # Applying sigmoid function to get probability

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)