Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame with columns named 'Feature_1', 'Feature_2', and 'target'.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = pd.DataFrame({
    'Feature_1': [0.059, 0.203, -0.016, -0.858, 1.634, -0.52, 1.909, 1.05, 0.121, 0.91, 0.39, -0.229, 0.226, -0.632, 0.075, -1.045, 0.38, -0.895, 1.87, 0.461, 1.01, -0.26, 1.989, -0.854, 1.976, 0.065, 0.109, -0.884, 0.28, 0.004, 2.007, -0.772, 0.395, 0.759, 0.207, 0.18, 0.748, 0.725, 0.426, -0.877, 1.995, -0.715, 0.83, -0.819, 0.02, -0.533, 2.029, -1.142, 1.655, 0.924, 0.797, -0.852, 1.941, 0.54, 1.749, -1.031, 1.152, -1.012, 0.097, 0.983, 0.308, -0.862, -0.048, 0.543, 1.636, 0.756, 0.299, -0.817, 1.644, 0.784, 1.983, 0.902, 0.279, 1.054, 0.089, 0.909, 1.477, -0.889, 0.007, 0.317, 0.163, -1.026, 1.785, 0.051, 0.557, -0.397, 1.704, -0.944, 1.481, -0.051, 0.587, 0.12, 0.163, -0.833, 0.815, -0.144, 0.25, -0.872, 1.419, 0.365],
    'Feature_2': [0.2, 0.944, 0.375, 0.617, -0.302, 0.975, 0.038, 0.149, 0.163, 0.419, -0.299, 0.959, -0.153, 0.769, 0.075, 0.185, -0.31, 0.425, 0.074, 0.849, -0.535, 0.915, 0.135, 0.397, 0.191, 0.977, 0.109, 0.339, -0.288, 0.972, 0.231, 0.586, -0.351, 0.736, -0.137, 1.003, -0.418, 0.651, -0.32, 0.431, 0.393, 0.661, -0.516, 0.587, 0.366, 0.785, 0.2, 0.108, -0.233, -0.07, -0.498, 0.441, 0.358, 0.88, -0.112, 0.264, -0.459, 0.256, -0.156, 0.037, -0.216, 0.547, 0.259, 0.873, -0.179, 0.621, -0.329, 0.594, -0.235, 0.639, 0.406, 0.093, -0.117, 0.123, 0.347, 0.283, -0.213, 0.372, 0.355, 0.838, -0.045, 0.248, 0.024, 0.993, -0.356, 0.937, -0.249, 0.516, -0.449, 0.966, -0.448, 1.003, -0.056, 0.452, -0.498, 0.959, -0.179, 0.544, -0.413, 0.983],
    'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Split the data into features and target
X = data[['Feature_1', 'Feature_2']]
y = data['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define the prediction function
def predict(x):
    df = x.copy()
    output = model.predict_proba(df[['Feature_1', 'Feature_2']])[:, 1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). It then trains a logistic regression model on this data. The predict function takes a DataFrame as input, makes a copy of it, and uses the trained model to predict the probability that the "target" is 1 for each row in the DataFrame. The output is an array of probabilities.