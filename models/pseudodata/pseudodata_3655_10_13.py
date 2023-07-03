import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Simple logic: if 'a' and 'c' are positive and 'b' and 'd' are negative, predict high probability for target 1
        # Otherwise, predict low probability for target 1
        if row['a'] > 0 and row['c'] > 0 and row['b'] < 0 and row['d'] < 0:
            y = 0.9
        else:
            y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)

# Test the function
data = {
    'a': [1.88, -0.02, 0.29, 0.51, 1.03, 1.01, 0.27, -0.89, 1.07, -0.72],
    'b': [-2.14, 0.35, -0.47, -0.35, -1.07, -1.11, 0.67, -0.44, -1.07, 0.39],
    'c': [1.56, 0.16, 0.18, 0.54, 0.91, 0.85, 0.72, -1.49, 0.96, -0.82],
    'd': [-1.52, -0.02, -0.22, -0.44, -0.85, -0.82, -0.32, 0.87, -0.88, 0.63],
    'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
print(predict(df))