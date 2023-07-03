Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code does not use any existing machine learning model, but rather implements the logistic regression model from scratch.

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
    w = np.array([0.1, 0.1])
    b = 0.1
    for index, row in df.iterrows():
        # Extract the features
        features = np.array([row['Feature_1'], row['Feature_2']])
        # Compute the probability
        y = logistic_regression(features, w, b)
        output.append(y)
    return np.array(output)
```

This code first defines the sigmoid function, which is used to map any real-valued number into the range [0, 1], and then defines the logistic regression model, which uses the sigmoid function to compute the probability that the "target" is 1 given the features. The predict function then applies the logistic regression model to each row of the input DataFrame, and returns an array of probabilities.

Please note that this code assumes that the weights and bias of the logistic regression model are known. In practice, these parameters would need to be learned from the data using a method such as gradient descent. However, this would require a more complex implementation and is beyond the scope of this task.