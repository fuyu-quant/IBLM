import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        feature_1 = row['Feature_1']
        feature_2 = row['Feature_2']

        # Calculate the distance from the origin (0, 0)
        distance = np.sqrt(feature_1**2 + feature_2**2)

        # Normalize the distance to a probability value between 0 and 1
        probability = 1 / (1 + np.exp(-distance))

        # Threshold for classification
        threshold = 0.5

        # Assign the probability value to y
        if probability > threshold:
            y = 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)