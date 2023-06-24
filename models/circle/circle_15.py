import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Based on the given data, we can observe that the target is 1 when the sum of Feature_1 and Feature_2 is positive
        # and the target is 0 when the sum is negative. We can use this observation to predict the probability of the target being 1.
        sum_features = feature_1 + feature_2
        y = max(0, min(1, (sum_features + 1) / 2))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)