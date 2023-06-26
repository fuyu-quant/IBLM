import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']
        
        # Use a simple linear combination of features to predict the probability
        y = 1 / (1 + np.exp(-(feature_1 + feature_2)))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)