import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Since we don't have any information about the relationship between the features and the target,
        # we will use a simple heuristic: if the sum of the absolute values of the features is greater than a threshold, predict 1, otherwise predict 0.
        # This is based on the observation that the features for target 1 tend to have larger absolute values than those for target 0.
        threshold = 1.0
        sum_abs_features = abs(row['Feature_1']) + abs(row['Feature_2'])
        if sum_abs_features > threshold:
            y = 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)