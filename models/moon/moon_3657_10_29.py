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
        # The coefficients for Feature_1 and Feature_2 are 0.4 and -0.6 respectively.
        # These coefficients are chosen such that the predicted value is close to 1 for target=1 and close to 0 for target=0.
        y = 0.4 * row['Feature_1'] - 0.6 * row['Feature_2']

        # The predicted value is then converted to a probability using the sigmoid function.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)