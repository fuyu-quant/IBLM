import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']
        
        # Custom logic to predict probability values based on the data
        probability = 1 / (1 + np.exp(-(feature_1 + feature_2)))
        
        # Do not change the code after this point.
        output.append(probability)
    return np.array(output)