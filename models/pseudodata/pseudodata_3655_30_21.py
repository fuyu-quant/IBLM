import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sum of the absolute values of the first four columns
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])

        # If the sum is greater than 2, predict a high probability (0.9) for target 1
        # Else, predict a low probability (0.1) for target 1
        y = 0.9 if sum_abs > 2 else 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)

# Test the function with some data
data = {
    'a': [1.88, -0.02, 0.29, 0.51, 1.03],
    'b': [-2.14, 0.35, -0.47, -0.35, -1.07],
    'c': [1.56, 0.16, 0.18, 0.54, 0.91],
    'd': [-1.52, -0.02, -0.22, -0.44, -0.85],
    'target': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
print(predict(df))