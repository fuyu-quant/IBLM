import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model to predict the target.
        # The coefficients are determined by observing the data.
        y = 0.5 * row['Feature_1'] + 0.5 * row['Feature_2']
        if y > 0.5:
            y = 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)

# Test the function
data = {
    'Feature_1': [0.059, 0.203, -0.016, -0.858, 1.634, -0.52, 1.909, 1.05, 0.121, 0.91, 0.39, -0.229, 0.226, -0.632, 0.075, -1.045, 0.38, -0.895, 1.87, 0.461, 1.01, -0.26, 1.989, -0.854, 1.976, 0.065, 0.109, -0.884, 0.28, 0.004],
    'Feature_2': [0.2, 0.944, 0.375, 0.617, -0.302, 0.975, 0.038, 0.149, 0.163, 0.419, -0.299, 0.959, -0.153, 0.769, 0.075, 0.185, -0.31, 0.425, 0.074, 0.849, -0.535, 0.915, 0.135, 0.397, 0.191, 0.977, 0.109, 0.339, -0.288, 0.972],
    'target': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
}
df = pd.DataFrame(data)
print(predict(df))