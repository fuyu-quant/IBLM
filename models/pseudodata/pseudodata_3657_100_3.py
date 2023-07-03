import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        # Here we are using a simple linear regression model for prediction
        # y = a*x1 + b*x2 + c*x3 + d*x4 + e
        # The coefficients a, b, c, d and e are determined based on the data
        a = 0.3
        b = 0.2
        c = -0.1
        d = 0.4
        e = 0.5
        y = a*row['a'] + b*row['b'] + c*row['c'] + d*row['d'] + e
        # Convert the output to a probability between 0 and 1 using the sigmoid function
        y = 1 / (1 + np.exp(-y))
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)