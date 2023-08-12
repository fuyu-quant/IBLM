import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple linear regression model to predict the target.
        # The coefficients are chosen such that the model predicts a high probability for target 1 and low probability for target 0.
        y = 1 / (1 + np.exp(-(0.5*row['Feature_1'] - 0.5*row['Feature_2'])))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)