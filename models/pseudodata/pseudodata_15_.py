import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the average of the first four columns (A, B, C, D)
        avg = (row['A'] + row['B'] + row['C'] + row['D']) / 4

        # Calculate the probability based on the average value
        y = 1 / (1 + np.exp(-avg))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)