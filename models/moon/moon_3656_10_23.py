import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple logic to predict the probability.
        # We are assuming that higher values of Feature_1 and lower values of Feature_2 are more likely to result in target 1.
        # This is based on the observation from the given data.
        # We normalize the features to the range [0, 1] and then calculate the probability as the average of the two features.
        # This is a very simple model and may not work well for more complex data.

        feature_1 = (row['Feature_1'] - df['Feature_1'].min()) / (df['Feature_1'].max() - df['Feature_1'].min())
        feature_2 = 1 - (row['Feature_2'] - df['Feature_2'].min()) / (df['Feature_2'].max() - df['Feature_2'].min())
        y = (feature_1 + feature_2) / 2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)