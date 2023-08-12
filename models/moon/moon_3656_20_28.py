import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple linear regression model to predict the target.
        # The coefficients are chosen based on the data provided.
        y = 0.5 * row['Feature_1'] - 0.4 * row['Feature_2'] + 0.6
        y = 1 / (1 + np.exp(-y))  # Apply sigmoid function to get probability

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)

# Test the function
data = {
    'Feature_1': [1.342, 2.029, 0.532, 0.021, 1.731, 0.753, 1.957, 1.209, 1.689, 0.06, 0.731, 0.256, 1.516, 0.749, 0.084, 1.643, 1.28, 1.998, 0.986, 0.095],
    'Feature_2': [-0.412, 0.302, -0.396, 0.333, -0.241, -0.613, 0.304, -0.53, -0.229, 0.525, -0.436, -0.106, -0.398, -0.513, 0.316, -0.379, -0.498, 0.356, -0.413, 0.099],
    'target': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
df = pd.DataFrame(data)
print(predict(df))