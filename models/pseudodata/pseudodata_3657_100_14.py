import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        # Here we are using a simple linear regression model for prediction
        # y = a*x1 + b*x2 + c*x3 + d*x4
        # We are assuming the coefficients a, b, c, d as 0.25, 0.25, 0.25, 0.25 respectively
        y = 0.25*row['a'] + 0.25*row['b'] + 0.25*row['c'] + 0.25*row['d']
        # Converting the output to probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)