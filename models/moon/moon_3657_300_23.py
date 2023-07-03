import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model to predict the target.
        # The coefficients are chosen based on the data distribution.
        y = 0.5 * row['Feature_1'] - 0.4 * row['Feature_2'] + 0.1

        # The output is a probability, so it should be between 0 and 1.
        # We use the sigmoid function to ensure this.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)