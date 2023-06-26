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

        # Adjust the probability based on the quadrant of the point
        if feature_1 > 0 and feature_2 > 0:
            probability *= 1.2
        elif feature_1 < 0 and feature_2 > 0:
            probability *= 0.8
        elif feature_1 < 0 and feature_2 < 0:
            probability *= 0.6
        else:
            probability *= 1.0

        # Clip the probability value between 0 and 1
        y = np.clip(probability, 0, 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)