import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Calculate the distance from the origin
        distance = np.sqrt(feature_1**2 + feature_2**2)

        # Set a threshold value for classification
        threshold = 0.8

        # Calculate the probability of the target being 1
        if distance < threshold:
            y = 1 - (distance / threshold)
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)