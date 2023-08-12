import numpy as np
import pandas as pd
from scipy.stats import norm

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the mean and standard deviation of each column
        mean_a = df['a'].mean()
        std_a = df['a'].std()
        mean_b = df['b'].mean()
        std_b = df['b'].std()
        mean_c = df['c'].mean()
        std_c = df['c'].std()
        mean_d = df['d'].mean()
        std_d = df['d'].std()

        # Calculate the probability of each column
        prob_a = norm.pdf(row['a'], mean_a, std_a)
        prob_b = norm.pdf(row['b'], mean_b, std_b)
        prob_c = norm.pdf(row['c'], mean_c, std_c)
        prob_d = norm.pdf(row['d'], mean_d, std_d)

        # Calculate the final probability
        y = prob_a * prob_b * prob_c * prob_d

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)