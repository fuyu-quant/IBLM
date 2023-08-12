import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the absolute values of the first four columns
        sum_abs = np.sum(np.abs(row[['a', 'b', 'c', 'd']]))

        # Calculate the mean of the first four columns
        mean = np.mean(row[['a', 'b', 'c', 'd']])

        # Calculate the standard deviation of the first four columns
        std = np.std(row[['a', 'b', 'c', 'd']])

        # Calculate the probability based on the sum of absolute values, mean and standard deviation
        y = (sum_abs + mean + std) / 3

        # Normalize the probability to be between 0 and 1
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)