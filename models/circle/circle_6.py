import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Based on the given data, it seems that the target is more likely to be 1 when the sum of the features is positive.
        # Therefore, we can use the sum of the features as a simple heuristic to predict the probability of the target being 1.
        sum_features = feature_1 + feature_2

        # Normalize the sum of features to a probability value between 0 and 1 using the sigmoid function.
        y = 1 / (1 + np.exp(-sum_features))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)