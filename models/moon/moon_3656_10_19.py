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
        # We are assuming m1 = 0.5, m2 = 0.4 and c = 0.1 for simplicity
        m1, m2, c = 0.5, 0.4, 0.1
        y = m1*row['Feature_1'] + m2*row['Feature_2'] + c

        # Converting the output to probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)