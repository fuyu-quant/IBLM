import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model for prediction.
        # The coefficients are calculated based on the given data.
        # The intercept is assumed to be 0 for simplicity.
        # The coefficients for Feature_1 and Feature_2 are 0.3 and -0.2 respectively.
        # These values are chosen based on the observation that Feature_1 has a positive correlation with the target,
        # and Feature_2 has a negative correlation with the target.
        y = 0.3 * row['Feature_1'] - 0.2 * row['Feature_2']

        # The output is then passed through a sigmoid function to convert it to a probability.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)