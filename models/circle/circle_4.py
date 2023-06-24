import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Based on the given data, it seems that the target is more likely to be 1 when the sum of Feature_1 and Feature_2 is positive.
        # We can use this observation to calculate the probability of the target being 1.
        probability = (feature_1 + feature_2 + 1) / 2

        # Clip the probability between 0 and 1 to avoid invalid values.
        y = np.clip(probability, 0, 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)