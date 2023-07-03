import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Since the target is binary (0 or 1), we can use a simple rule-based approach to predict the probability.
        # From the given data, it seems that when Feature_1 is high and Feature_2 is low, the target is more likely to be 1.
        # We can create a simple rule that calculates the probability based on these observations.
        # This is a very basic approach and may not work well with more complex data.
        
        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']
        
        # Normalize the features to be between 0 and 1
        feature_1 = (feature_1 - df['Feature_1'].min()) / (df['Feature_1'].max() - df['Feature_1'].min())
        feature_2 = (feature_2 - df['Feature_2'].min()) / (df['Feature_2'].max() - df['Feature_2'].min())
        
        # Calculate the probability as the average of the two features
        # This gives a high probability when Feature_1 is high and Feature_2 is low
        y = (feature_1 + (1 - feature_2)) / 2
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)