Here is a simple Python code that uses a basic logistic regression model to predict the probability that the "target" of the unknown data is 1. This code does not use any existing machine learning model, but rather implements the logistic regression model from scratch.

```python
import numpy as np
import pandas as pd

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the logistic regression model
def logistic_regression(x, weights):
    return sigmoid(np.dot(x, weights))

# Define the function to predict the probability
def predict(x):
    df = x.copy()
    output = []
    # Initialize the weights
    weights = np.zeros(df.shape[1] - 1)
    # Iterate over the rows of the dataframe
    for index, row in df.iterrows():
        # Extract the features and the target
        features = row[:-1]
        target = row[-1]
        # Compute the prediction
        prediction = logistic_regression(features, weights)
        # Update the weights
        weights += 0.01 * (target - prediction) * features
        # Append the prediction to the output
        output.append(prediction)
    return np.array(output)
```

This code first defines the sigmoid function, which is used in the logistic regression model to map any real-valued number into the range [0, 1]. Then, it defines the logistic regression model, which computes the dot product of the features and the weights, and applies the sigmoid function to the result.

The `predict` function initializes the weights to zero, and then iterates over the rows of the dataframe. For each row, it extracts the features and the target, computes the prediction using the logistic regression model, updates the weights using the gradient descent algorithm, and appends the prediction to the output.

Finally, the `predict` function returns the output as a numpy array.