import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Based on the given data, we can observe that the target is more likely to be 1 when Feature_1 is positive
        # and Feature_2 is negative or close to 0. Similarly, the target is more likely to be 0 when Feature_1 is
        # close to 0 and Feature_2 is positive. We can use this observation to predict the probability of the target
        # being 1.

        # Calculate the probability of target being 1 based on Feature_1 and Feature_2
        prob_1 = (1 / (1 + np.exp(-feature_1))) * (1 - (1 / (1 + np.exp(-feature_2))))

        # Do not change the code after this point.
        output.append(prob_1)
    return np.array(output)