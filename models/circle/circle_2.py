import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Based on the given data, we can observe that the target is more likely to be 1 when
        # the absolute values of Feature_1 and Feature_2 are higher.
        # We can use a simple heuristic to predict the probability based on the sum of the absolute values.
        probability = (abs(feature_1) + abs(feature_2)) / 2

        # Normalize the probability to be between 0 and 1
        probability = min(max(probability, 0), 1)

        # Do not change the code after this point.
        output.append(probability)
    return np.array(output)