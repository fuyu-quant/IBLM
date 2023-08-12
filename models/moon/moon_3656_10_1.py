import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model to predict the target.
        # The coefficients of the model are determined by analyzing the given data.
        # The intercept is set to 0.5 to ensure that the predicted probabilities are between 0 and 1.
        y = 0.5 + 0.3 * row['Feature_1'] - 0.2 * row['Feature_2']

        # The predicted probability is then clipped to the range [0, 1] to ensure it is a valid probability.
        y = np.clip(y, 0, 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)