import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the weighted sum of the features
        weighted_sum = row['a'] * 0.3 + row['b'] * 0.2 + row['c'] * 0.4 + row['d'] * 0.1

        # Apply the sigmoid function to the weighted sum to get the probability
        y = 1 / (1 + np.exp(-weighted_sum))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)