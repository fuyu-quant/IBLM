import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the absolute values of the columns a, b, c, d
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])

        # Calculate the sum of the columns a, b, c, d
        sum_val = row['a'] + row['b'] + row['c'] + row['d']

        # Calculate the average of the columns a, b, c, d
        avg_val = sum_val / 4

        # Calculate the probability based on the sum of absolute values, sum of values and average
        y = (sum_abs + sum_val + avg_val) / 3

        # Normalize the probability to be between 0 and 1
        y = (y - df.min().min()) / (df.max().max() - df.min().min())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)