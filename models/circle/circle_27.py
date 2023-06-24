import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Based on the given data, it seems that the target is more likely to be 1 when the sum of the absolute values of the features is less than 1.
        # Therefore, we can use this observation to predict the probability of the target being 1.
        sum_abs_features = abs(feature_1) + abs(feature_2)
        if sum_abs_features < 1:
            y = 1 - sum_abs_features
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)