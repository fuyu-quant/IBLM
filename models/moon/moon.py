import numpy as np
import pandas as pd
from scipy.stats import norm

def predict(x):
    df = x.copy()
    output = []
    
    # Calculate mean and standard deviation for each feature for each class
    mean_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].mean()
    std_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].std()
    mean_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].mean()
    std_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].std()
    
    for index, row in df.iterrows():
        # Calculate the probability of the data point for each class
        prob_0 = norm.pdf(row['Feature_1'], mean_0['Feature_1'], std_0['Feature_1']) * \
                  norm.pdf(row['Feature_2'], mean_0['Feature_2'], std_0['Feature_2'])
        prob_1 = norm.pdf(row['Feature_1'], mean_1['Feature_1'], std_1['Feature_1']) * \
                  norm.pdf(row['Feature_2'], mean_1['Feature_2'], std_1['Feature_2'])
        
        # Normalize the probabilities so they sum to 1
        total_prob = prob_0 + prob_1
        prob_0 /= total_prob
        prob_1 /= total_prob
        
        # Append the probability of class 1 to the output
        output.append(prob_1)
    
    return np.array(output)