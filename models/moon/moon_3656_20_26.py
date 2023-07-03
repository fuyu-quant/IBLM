import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model for prediction.
        # The coefficients are estimated based on the given data.
        # The intercept is assumed to be 0 for simplicity.
        # The model is: y = 0.5*Feature_1 + 0.5*Feature_2
        y = 0.5*row['Feature_1'] + 0.5*row['Feature_2']

        # Convert the output to a probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)