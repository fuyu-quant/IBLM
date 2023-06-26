import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on Feature_1 and Feature_2
        prob = 1 / (1 + np.exp(-(row['Feature_1'] + row['Feature_2'])))
        
        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)