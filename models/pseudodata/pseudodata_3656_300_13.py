import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the first four columns
        sum_of_values = row['a'] + row['b'] + row['c'] + row['d']

        # Normalize the sum to a range between 0 and 1
        normalized_sum = (sum_of_values + 10) / 20

        # Use the normalized sum as the predicted probability
        y = normalized_sum

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)