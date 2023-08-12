Here is a simple Python code that uses a basic logistic regression model to predict the probability that the "target" of the unknown data is 1. This code does not use any existing machine learning model, but rather implements the logistic regression model from scratch.

```python
import numpy as np
import pandas as pd

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the logistic regression model
def logistic_regression(X, y, num_iterations, learning_rate):
    n = X.shape[1]
    weights = np.zeros((n, 1))
    bias = 0

    for i in range(num_iterations):
        z = np.dot(X, weights) + bias
        p = sigmoid(z)

        gradient_weights = np.dot(X.T, (p - y)) / y.size
        gradient_bias = np.sum(p - y) / y.size

        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return weights, bias

# Define the prediction function
def predict(x):
    df = x.copy()
    output = []

    # Convert the DataFrame to a NumPy array
    X = df[['Feature_1', 'Feature_2']].values
    y = df['target'].values.reshape(-1, 1)

    # Train the logistic regression model
    weights, bias = logistic_regression(X, y, num_iterations=1000, learning_rate=0.01)

    for index, row in df.iterrows():
        # Calculate the probability that the target is 1
        z = np.dot(row[['Feature_1', 'Feature_2']].values, weights) + bias
        y = sigmoid(z)

        output.append(y[0])

    return np.array(output)
```

This code first defines the sigmoid function, which is used in the logistic regression model to map any real-valued number into the range [0, 1]. Then it defines the logistic regression model, which iteratively updates the weights and bias to minimize the difference between the predicted probabilities and the actual target values. Finally, it defines the prediction function, which uses the trained logistic regression model to predict the probability that the target is 1 for each row in the input DataFrame.