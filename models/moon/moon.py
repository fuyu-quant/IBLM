import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the distance from the origin (0, 0)
        distance = np.sqrt(row['Feature_1']**2 + row['Feature_2']**2)

        # Normalize the distance to a probability value between 0 and 1
        y = 1 / (1 + np.exp(-distance))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)