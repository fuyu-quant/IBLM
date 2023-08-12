import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the absolute values of the first four columns
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])

        # Normalize the sum to a range of 0 to 1
        normalized_sum = sum_abs / 4

        # If the normalized sum is greater than 0.5, predict a high probability for target 1
        # Otherwise, predict a low probability for target 1
        if normalized_sum > 0.5:
            y = 1 - normalized_sum
        else:
            y = normalized_sum

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)