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

        # Calculate the sum of the positive values of the features
        sum_pos = max(0, row['a']) + max(0, row['b']) + max(0, row['c']) + max(0, row['d'])

        # Calculate the probability as the ratio of the sum of positive values to the sum of absolute values
        y = sum_pos / sum_abs if sum_abs != 0 else 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)