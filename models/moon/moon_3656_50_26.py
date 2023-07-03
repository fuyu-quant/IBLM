import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the mean of Feature_1 and Feature_2
        mean_feature = (row['Feature_1'] + row['Feature_2']) / 2

        # If the mean of Feature_1 and Feature_2 is greater than 0, predict a high probability for target 1
        if mean_feature > 0:
            y = 1
        # If the mean of Feature_1 and Feature_2 is less than or equal to 0, predict a low probability for target 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)

# Test the function with some data
data = {
    'Feature_1': [1.342, 2.029, 0.532, 0.021, 1.731],
    'Feature_2': [-0.412, 0.302, -0.396, 0.333, -0.241],
    'target': [1.0, 1.0, 1.0, 1.0, 1.0]
}
df = pd.DataFrame(data)
print(predict(df))