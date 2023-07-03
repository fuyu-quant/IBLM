import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple heuristic to predict the target.
        # We are assuming that if the sum of the values in the row is positive, the target is likely to be 1.
        # If the sum of the values in the row is negative, the target is likely to be 0.
        # We then normalize these sums to be between 0 and 1 to represent probabilities.
        sum_row = row['a'] + row['b'] + row['c'] + row['d']
        if sum_row > 0:
            y = sum_row / (sum_row + abs(df[['a', 'b', 'c', 'd']].values.min()))
        else:
            y = sum_row / (sum_row - abs(df[['a', 'b', 'c', 'd']].values.max()))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)