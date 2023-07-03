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
        # The intercept is set to 0.5 to make sure the output is between 0 and 1.
        y = 0.5 + 0.3 * row['Feature_1'] - 0.2 * row['Feature_2']

        # Make sure the output is between 0 and 1.
        y = max(0, min(1, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)