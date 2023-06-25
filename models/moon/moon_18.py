import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the distance from the origin (0, 0)
        distance = np.sqrt(row['Feature_1']**2 + row['Feature_2']**2)

        # Normalize the distance to a value between 0 and 1
        normalized_distance = distance / (distance + 1)

        # Use the normalized distance as the probability of the target being 1
        y = normalized_distance

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)