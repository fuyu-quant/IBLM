import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple heuristic to predict the target.
        # If Feature_1 is greater than 0 and Feature_2 is less than 0.5, we predict a high probability for target 1.
        # Otherwise, we predict a low probability for target 1.
        if row['Feature_1'] > 0 and row['Feature_2'] < 0.5:
            y = 0.9
        else:
            y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)

# Example usage:
# df = pd.DataFrame({
#     'Feature_1': [0.059, 0.203, -0.016, -0.858, 1.634],
#     'Feature_2': [0.2, 0.944, 0.375, 0.617, -0.302],
#     'target': [1.0, 0.0, 1.0, 0.0, 1.0]
# })
# print(predict(df))