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
        # We are assuming m = 0.5 and c = 0.5 for simplicity
        y = 0.5 * row['Feature_1'] + 0.5 * row['Feature_2'] + 0.5

        # Since we are predicting probabilities, the output should be between 0 and 1
        # We use the sigmoid function to ensure this
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)