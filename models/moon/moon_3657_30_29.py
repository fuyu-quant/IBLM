import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the mean of Feature_1 and Feature_2
        mean_feature = (row['Feature_1'] + row['Feature_2']) / 2

        # If the mean of Feature_1 and Feature_2 is greater than 0, the probability of target being 1 is high.
        # If the mean of Feature_1 and Feature_2 is less than 0, the probability of target being 1 is low.
        if mean_feature > 0:
            y = 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)