import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the mean of Feature_1 and Feature_2
        mean_value = (row['Feature_1'] + row['Feature_2']) / 2

        # If the mean value is greater than 0, the probability of target being 1 is high
        if mean_value > 0:
            y = 1
        # If the mean value is less than or equal to 0, the probability of target being 1 is low
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)

# Test the function
data = {
    'Feature_1': [0.059, 0.203, -0.016, -0.858, 1.634, -0.52, 1.909, 1.05, 0.121, 0.91, 0.39, -0.229, 0.226, -0.632, 0.075, -1.045, 0.38, -0.895, 1.87, 0.461],
    'Feature_2': [0.2, 0.944, 0.375, 0.617, -0.302, 0.975, 0.038, 0.149, 0.163, 0.419, -0.299, 0.959, -0.153, 0.769, 0.075, 0.185, -0.31, 0.425, 0.074, 0.849],
    'target': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
}
df = pd.DataFrame(data)
print(predict(df))