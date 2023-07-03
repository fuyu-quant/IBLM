import numpy as np
import pandas as pd
from scipy.stats import norm

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Calculate the mean and standard deviation of each feature for target 0 and 1
        mean_0 = df[df['target'] == 0].mean()
        std_0 = df[df['target'] == 0].std()
        mean_1 = df[df['target'] == 1].mean()
        std_1 = df[df['target'] == 1].std()

        # Calculate the probability of each feature given target 0 and 1 using Gaussian distribution
        prob_0 = norm.pdf(row[:-1], mean_0[:-1], std_0[:-1]).prod()
        prob_1 = norm.pdf(row[:-1], mean_1[:-1], std_1[:-1]).prod()

        # Calculate the prior probability of target 0 and 1
        prior_0 = len(df[df['target'] == 0]) / len(df)
        prior_1 = len(df[df['target'] == 1]) / len(df)

        # Calculate the posterior probability of target 0 and 1
        post_0 = prob_0 * prior_0
        post_1 = prob_1 * prior_1

        # Normalize the posterior probabilities to get the final prediction
        y = post_1 / (post_0 + post_1)
        output.append(y)
    return np.array(output)