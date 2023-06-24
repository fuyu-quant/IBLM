import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Simple linear combination of features to predict the probability
        y = 0.5 * feature_1 + 0.5 * feature_2

        # Normalize the probability to be between 0 and 1
        y = (y + 1) / 2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)