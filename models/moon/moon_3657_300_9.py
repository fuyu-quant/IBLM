import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model to predict the target.
        # The coefficients are chosen such that the model predicts a high probability for target=1 and a low probability for target=0.
        # The coefficients are chosen based on the observation that Feature_1 is generally higher for target=1 and Feature_2 is generally lower for target=1.
        y = 0.5 + 0.3 * row['Feature_1'] - 0.3 * row['Feature_2']

        # The predicted value is then passed through a sigmoid function to ensure it lies between 0 and 1.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)