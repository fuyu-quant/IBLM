import numpy as np
import pandas as pd
from scipy.stats import norm

def predict(x):
    df = x.copy()
    output = []
    
    # Separate the data into two groups based on the target value
    group_1 = df[df['target'] == 1]
    group_0 = df[df['target'] == 0]
    
    # Calculate the mean and standard deviation for each feature in each group
    mean_1 = group_1[['Feature_1', 'Feature_2']].mean()
    std_1 = group_1[['Feature_1', 'Feature_2']].std()
    mean_0 = group_0[['Feature_1', 'Feature_2']].mean()
    std_0 = group_0[['Feature_1', 'Feature_2']].std()
    
    for index, row in df.iterrows():
        # Calculate the probability density for each feature in each group
        prob_1 = norm.pdf(row['Feature_1'], mean_1['Feature_1'], std_1['Feature_1']) * \
                  norm.pdf(row['Feature_2'], mean_1['Feature_2'], std_1['Feature_2'])
        prob_0 = norm.pdf(row['Feature_1'], mean_0['Feature_1'], std_0['Feature_1']) * \
                  norm.pdf(row['Feature_2'], mean_0['Feature_2'], std_0['Feature_2'])
        
        # Calculate the probability that the target is 1
        y = prob_1 / (prob_1 + prob_0)
        
        output.append(y)
    
    return np.array(output)