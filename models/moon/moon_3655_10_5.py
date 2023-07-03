import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the mean of Feature_1 and Feature_2 for target 0 and 1
        mean_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].mean().values
        mean_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].mean().values

        # Calculate the standard deviation of Feature_1 and Feature_2 for target 0 and 1
        std_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].std().values
        std_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].std().values

        # Calculate the probability of the data point belonging to target 0 and 1 using Gaussian distribution
        prob_0 = (1 / (np.sqrt(2 * np.pi * std_0**2))) * np.exp(-((row[['Feature_1', 'Feature_2']].values - mean_0)**2 / (2 * std_0**2)))
        prob_1 = (1 / (np.sqrt(2 * np.pi * std_1**2))) * np.exp(-((row[['Feature_1', 'Feature_2']].values - mean_1)**2 / (2 * std_1**2)))

        # Normalize the probabilities to get the final prediction
        y = prob_1 / (prob_0 + prob_1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)