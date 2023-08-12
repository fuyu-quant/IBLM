Here is a simple Python code that uses a basic logistic regression model to predict the probability that the "target" of the unknown data is 1. This code does not use any existing machine learning model, but rather implements the logistic regression model from scratch.

```python
import numpy as np
import pandas as pd

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the logistic regression model
def logistic_regression(x, w):
    return sigmoid(np.dot(x, w))

# Define the predict function
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Extract the features
        features = np.array([row['Feature_1'], row['Feature_2'], 1])
        # Define the weights (these should be learned from the data)
        weights = np.array([0.5, 0.5, 0])
        # Compute the prediction
        y = logistic_regression(features, weights)
        output.append(y)
    return np.array(output)

# Test the predict function
data = {
    'Feature_1': [0.059, 0.203, -0.016, -0.858, 1.634, -0.52, 1.909, 1.05, 0.121, 0.91],
    'Feature_2': [0.2, 0.944, 0.375, 0.617, -0.302, 0.975, 0.038, 0.149, 0.163, 0.419],
    'target': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
}
df = pd.DataFrame(data)
print(predict(df))
```

Please note that this code uses a fixed set of weights for the logistic regression model. In a real-world scenario, these weights should be learned from the data using a learning algorithm.