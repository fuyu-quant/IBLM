import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the absolute values of the features
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])

        # Calculate the sum of the features
        sum_features = row['a'] + row['b'] + row['c'] + row['d']

        # Calculate the average of the features
        avg_features = sum_features / 4

        # Calculate the probability
        y = (sum_abs + avg_features) / 2

        # Normalize the probability to be between 0 and 1
        y = (y - min(y, 0)) / (max(y, 0) - min(y, 0))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)