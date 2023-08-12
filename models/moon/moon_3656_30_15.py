import numpy as np
import pandas as pd
from scipy.stats import norm

def predict(x):
    df = x.copy()
    output = []
    
    # Calculate the mean and standard deviation of the features for target 1
    mean_feature_1 = df[df['target'] == 1]['Feature_1'].mean()
    std_feature_1 = df[df['target'] == 1]['Feature_1'].std()
    
    mean_feature_2 = df[df['target'] == 1]['Feature_2'].mean()
    std_feature_2 = df[df['target'] == 1]['Feature_2'].std()
    
    for index, row in df.iterrows():
        # Calculate the probability of the feature values given target 1 using Gaussian distribution
        prob_feature_1_given_target_1 = norm.pdf(row['Feature_1'], mean_feature_1, std_feature_1)
        prob_feature_2_given_target_1 = norm.pdf(row['Feature_2'], mean_feature_2, std_feature_2)
        
        # The final probability is the product of the individual probabilities
        y = prob_feature_1_given_target_1 * prob_feature_2_given_target_1
        
        output.append(y)
        
    return np.array(output)