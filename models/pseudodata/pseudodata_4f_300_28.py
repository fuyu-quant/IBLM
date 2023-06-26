import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        a, b, c, d = row['a'], row['b'], row['c'], row['d']
        
        # Calculate the weighted sum of the features
        weighted_sum = a * 0.3 + b * 0.2 + c * 0.4 + d * 0.1
        
        # Apply the sigmoid function to the weighted sum to get the probability
        y = 1 / (1 + np.exp(-weighted_sum))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)