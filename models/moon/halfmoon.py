import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, it seems that the target is more likely to be 1 when Feature_1 is positive
        # and Feature_2 is negative or close to 0. Conversely, the target is more likely to be 0 when
        # Feature_1 is negative or close to 0 and Feature_2 is positive.
        # We can use this observation to create a simple heuristic for predicting the probability of the target being 1.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Calculate the probability of the target being 1 based on the heuristic
        prob_1 = (feature_1 + 1) / 2 * (1 - (feature_2 + 1) / 2)

        # Do not change the code after this point.
        output.append(prob_1)
    return np.array(output)