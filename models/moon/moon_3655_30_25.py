import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple rule-based approach to predict the target.
        # If Feature_1 is greater than 0 and Feature_2 is less than 0.5, we predict a high probability for target 1.
        # If Feature_1 is less than 0 or Feature_2 is greater than 0.5, we predict a low probability for target 1.
        # This rule is based on the observation from the given data.
        if row['Feature_1'] > 0 and row['Feature_2'] < 0.5:
            y = 0.9
        else:
            y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)