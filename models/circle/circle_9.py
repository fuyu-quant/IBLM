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
        # the absolute value of Feature_1 is greater than 0.5 and the absolute value of Feature_2 is less than 0.5
        # or when the absolute value of Feature_1 is less than 0.5 and the absolute value of Feature_2 is greater than 0.5.
        # We can use this observation to estimate the probability of the target being 1.

        if (abs(feature_1) > 0.5 and abs(feature_2) < 0.5) or (abs(feature_1) < 0.5 and abs(feature_2) > 0.5):
            y = 0.9
        else:
            y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)