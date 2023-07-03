import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple logic to predict the probability.
        # If Feature_1 is greater than 1 and Feature_2 is less than 0, we predict high probability for target 1.
        # If Feature_1 is less than 1 and Feature_2 is greater than 0, we predict low probability for target 1.
        # For all other cases, we predict a medium probability for target 1.
        if row['Feature_1'] > 1 and row['Feature_2'] < 0:
            y = 0.9
        elif row['Feature_1'] < 1 and row['Feature_2'] > 0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)