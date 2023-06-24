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
        # the sum of the absolute values of Feature_1 and Feature_2 is less than 1.
        # Therefore, we can use this observation to predict the probability of the target being 1.
        probability = 1 - abs(feature_1 + feature_2)

        # Do not change the code after this point.
        output.append(probability)
    return np.array(output)