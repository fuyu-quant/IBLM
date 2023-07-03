import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Simple logic: if the sum of a, b, c, and d is positive, predict a high probability for target 1.
        # If the sum is negative, predict a low probability for target 1.
        sum_row = row['a'] + row['b'] + row['c'] + row['d']
        if sum_row > 0:
            y = 0.9  # High probability for target 1
        else:
            y = 0.1  # Low probability for target 1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)

# Example usage:
data = {
    'a': [1.88, -0.02, 0.29, 0.51, 1.03],
    'b': [-2.14, 0.35, -0.47, -0.35, -1.07],
    'c': [1.56, 0.16, 0.18, 0.54, 0.91],
    'd': [-1.52, -0.02, -0.22, -0.44, -0.85],
    'target': [1.0, 0.0, 1.0, 0.0, 1.0]
}
df = pd.DataFrame(data)
print(predict(df))