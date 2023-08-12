import numpy as np
import pandas as pd
from scipy.stats import norm

def predict(x):
    df = x.copy()
    output = []
    
    # Calculate mean and standard deviation for each feature for target 0 and 1
    mean_1_0 = df[df['target'] == 0]['Feature_1'].mean()
    std_1_0 = df[df['target'] == 0]['Feature_1'].std()
    mean_2_0 = df[df['target'] == 0]['Feature_2'].mean()
    std_2_0 = df[df['target'] == 0]['Feature_2'].std()
    
    mean_1_1 = df[df['target'] == 1]['Feature_1'].mean()
    std_1_1 = df[df['target'] == 1]['Feature_1'].std()
    mean_2_1 = df[df['target'] == 1]['Feature_2'].mean()
    std_2_1 = df[df['target'] == 1]['Feature_2'].std()
    
    for index, row in df.iterrows():
        # Calculate the probability of the data point for each feature for target 0 and 1
        prob_1_0 = norm.pdf(row['Feature_1'], mean_1_0, std_1_0)
        prob_2_0 = norm.pdf(row['Feature_2'], mean_2_0, std_2_0)
        prob_1_1 = norm.pdf(row['Feature_1'], mean_1_1, std_1_1)
        prob_2_1 = norm.pdf(row['Feature_2'], mean_2_1, std_2_1)
        
        # Calculate the total probability for target 0 and 1
        prob_0 = prob_1_0 * prob_2_0
        prob_1 = prob_1_1 * prob_2_1
        
        # Normalize the probabilities so they sum to 1
        total_prob = prob_0 + prob_1
        prob_0 /= total_prob
        prob_1 /= total_prob
        
        # The output is the probability that the target is 1
        output.append(prob_1)
    
    return np.array(output)