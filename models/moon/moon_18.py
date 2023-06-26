import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given data
        probability = 1 / (1 + np.exp(-(row['Feature_1'] + row['Feature_2'])))

        # Do not change the code after this point.
        output.append(probability)
    return np.array(output)