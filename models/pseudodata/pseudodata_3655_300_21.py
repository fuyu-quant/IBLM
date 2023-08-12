import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        # Here we are using a simple heuristic to predict the target.
        # If the sum of the absolute values of the first four columns is greater than a threshold, we predict 1, otherwise we predict 0.
        # This is a very simple heuristic and may not give good results on real data.
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])
        if sum_abs > 2:
            y = 1
        else:
            y = 0
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)