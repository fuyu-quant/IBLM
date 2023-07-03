import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the absolute values of the first four columns
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])

        # If the sum is greater than 2, predict a high probability for target 1
        if sum_abs > 2:
            y = 0.9
        # If the sum is less than or equal to 2, predict a low probability for target 1
        else:
            y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)