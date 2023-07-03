import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model for prediction
        # The coefficients are calculated based on the given data
        # The equation of the model is y = 0.5*Feature_1 - 0.3*Feature_2 + 0.2
        y = 0.5*row['Feature_1'] - 0.3*row['Feature_2'] + 0.2

        # Converting the output to probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)