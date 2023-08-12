import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        # Here we are using a simple heuristic to predict the target.
        # If the sum of the values in the row is positive, we predict a high probability for target 1.
        # If the sum of the values in the row is negative, we predict a low probability for target 1.
        row_sum = row['a'] + row['b'] + row['c'] + row['d']
        if row_sum > 0:
            y = 0.9
        else:
            y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)