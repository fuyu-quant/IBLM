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
        # The coefficients for Feature_1 and Feature_2 are 0.5 and -0.5 respectively.
        # These coefficients are chosen to give a high probability for target 1 and low probability for target 0.
        # The sigmoid function is used to convert the linear regression output to a probability between 0 and 1.

        y = 1 / (1 + np.exp(-(0.5 * row['Feature_1'] - 0.5 * row['Feature_2'])))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)