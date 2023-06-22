import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the weighted sum of the features
        weighted_sum = row['A'] * 0.3 + row['B'] * 0.2 + row['C'] * 0.4 + row['D'] * 0.1

        # Apply a sigmoid function to the weighted sum to get the probability
        y = 1 / (1 + np.exp(-weighted_sum))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)