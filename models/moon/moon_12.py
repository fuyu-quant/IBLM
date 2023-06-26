import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the distance from the origin
        distance = np.sqrt(row['Feature_1']**2 + row['Feature_2']**2)
        
        # Normalize the distance to a probability value between 0 and 1
        probability = 1 / (1 + np.exp(-distance))
        
        # Set a threshold to classify the data
        threshold = 0.5
        if probability > threshold:
            y = 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)