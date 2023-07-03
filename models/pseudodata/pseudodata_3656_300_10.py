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

        # Calculate the standard deviation of the first four columns
        std = np.std([row['a'], row['b'], row['c'], row['d']])

        # Calculate the prediction as a weighted sum of the sum_abs, mean and std
        y = 0.5 * sum_abs + 0.3 * mean + 0.2 * std

        # Normalize the prediction to be between 0 and 1
        y = (y - df[['a', 'b', 'c', 'd']].values.min()) / (df[['a', 'b', 'c', 'd']].values.max() - df[['a', 'b', 'c', 'd']].values.min())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)