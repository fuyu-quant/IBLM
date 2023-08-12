import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model for prediction
        # y = mx + c, where m is the slope and c is the y-intercept
        # We are assuming that the slope (m) is 1 and y-intercept (c) is 0 for simplicity
        # This is a very basic model and may not give accurate results for complex datasets

        y = row['Feature_1'] + row['Feature_2']
        if y > 1:
            y = 1
        elif y < 0:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)