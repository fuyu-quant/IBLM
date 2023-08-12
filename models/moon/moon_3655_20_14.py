Here is a simple Python code that uses a basic logistic regression model to predict the probability that the "target" of the unknown data is 1. This code does not use any existing machine learning model, but rather implements the logistic regression model from scratch.

```python
import numpy as np
import pandas as pd

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the logistic regression model
def logistic_regression(x, w, b):
    return sigmoid(np.dot(x, w) + b)

# Define the predict function
def predict(x):
    df = x.copy()
    output = []
    # Initialize the weights and bias
    w = np.array([0.1, 0.2])
    b = 0.1
    for index, row in df.iterrows():
        # Extract the features
        features = np.array([row['Feature_1'], row['Feature_2']])
        # Compute the probability
        y = logistic_regression(features, w, b)
        output.append(y)
    return np.array(output)

# Test the predict function
data = pd.DataFrame({
    'Feature_1': [0.059, 0.203, -0.016, -0.858, 1.634, -0.52, 1.909, 1.05, 0.121, 0.91, 0.39, -0.229, 0.226, -0.632, 0.075, -1.045, 0.38, -0.895, 1.87, 0.461],
    'Feature_2': [0.2, 0.944, 0.375, 0.617, -0.302, 0.975, 0.038, 0.149, 0.163, 0.419, -0.299, 0.959, -0.153, 0.769, 0.075, 0.185, -0.31, 0.425, 0.074, 0.849],
    'target': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
})
print(predict(data))
```

Please note that this code is a very basic implementation of logistic regression and does not include any optimization of the weights and bias. In a real-world scenario, you would typically use a more sophisticated method to train the logistic regression model, such as gradient descent or a more advanced optimization algorithm.