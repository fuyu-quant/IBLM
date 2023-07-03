import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model to predict the target.
        # The coefficients are chosen based on the observation of the data.
        y = 0.5 * row['Feature_1'] + 0.5 * row['Feature_2']
        y = 1 / (1 + np.exp(-y))  # Apply sigmoid function to get probability

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)