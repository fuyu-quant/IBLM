import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the absolute values of the first four columns
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])

        # Calculate the sum of the squares of the first four columns
        sum_squares = row['a']**2 + row['b']**2 + row['c']**2 + row['d']**2

        # Calculate the product of the first four columns
        product = row['a'] * row['b'] * row['c'] * row['d']

        # Calculate the average of the first four columns
        average = (row['a'] + row['b'] + row['c'] + row['d']) / 4

        # Calculate the prediction as a weighted sum of the above four values
        y = 0.2 * sum_abs + 0.3 * sum_squares + 0.1 * product + 0.4 * average

        # Normalize the prediction to the range [0, 1]
        y = (y - df.iloc[:, :-1].values.min()) / (df.iloc[:, :-1].values.max() - df.iloc[:, :-1].values.min())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)