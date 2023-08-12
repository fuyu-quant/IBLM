import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the absolute values of the first four columns
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])

        # Calculate the mean of the first four columns
        mean = (row['a'] + row['b'] + row['c'] + row['d']) / 4

        # If the sum of the absolute values is greater than 1 and the mean is less than 0, predict a high probability for target 1
        if sum_abs > 1 and mean < 0:
            y = 0.9
        # If the sum of the absolute values is less than or equal to 1 and the mean is greater than or equal to 0, predict a low probability for target 1
        elif sum_abs <= 1 and mean >= 0:
            y = 0.1
        # In other cases, predict a medium probability for target 1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)