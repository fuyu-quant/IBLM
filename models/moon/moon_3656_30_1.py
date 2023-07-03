import numpy as np
import pandas as pd
from scipy.stats import norm

def predict(x):
    df = x.copy()
    output = []
    
    # Calculate mean and standard deviation for each feature for target 1
    mean_feature_1 = df[df['target'] == 1]['Feature_1'].mean()
    std_feature_1 = df[df['target'] == 1]['Feature_1'].std()
    
    mean_feature_2 = df[df['target'] == 1]['Feature_2'].mean()
    std_feature_2 = df[df['target'] == 1]['Feature_2'].std()
    
    for index, row in df.iterrows():
        # Calculate the probability of the data point for each feature using Gaussian distribution
        prob_feature_1 = norm.pdf(row['Feature_1'], mean_feature_1, std_feature_1)
        prob_feature_2 = norm.pdf(row['Feature_2'], mean_feature_2, std_feature_2)
        
        # Multiply the probabilities of each feature to get the final probability
        y = prob_feature_1 * prob_feature_2
        
        output.append(y)
        
    return np.array(output)