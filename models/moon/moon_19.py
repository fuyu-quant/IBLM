import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the distance from the center of each class
        distance_1 = np.sqrt((row['Feature_1'] - 1)**2 + (row['Feature_2'] - 0)**2)
        distance_0 = np.sqrt((row['Feature_1'] - 0)**2 + (row['Feature_2'] - 1)**2)
        
        # Calculate the probability of belonging to class 1 using the softmax function
        y = np.exp(-distance_1) / (np.exp(-distance_1) + np.exp(-distance_0))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)