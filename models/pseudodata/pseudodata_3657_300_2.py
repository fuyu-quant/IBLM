import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple heuristic to predict the target.
        # We are assuming that if the sum of the values in the row is positive, the target is likely to be 1.
        # If the sum is negative, the target is likely to be 0.
        # We then normalize this sum to be between 0 and 1 to represent a probability.
        row_sum = row['a'] + row['b'] + row['c'] + row['d']
        if row_sum > 0:
            y = row_sum / (row_sum + abs(row_sum))
        else:
            y = row_sum / (row_sum - abs(row_sum))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)