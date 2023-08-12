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
    # Initialize the weights
    w = np.array([0.1, 0.1])
    # Iterate over the rows of the dataframe
    for index, row in df.iterrows():
        # Extract the features
        features = np.array([row['Feature_1'], row['Feature_2']])
        # Compute the prediction
        y = logistic_regression(features, w)
        # Append the prediction to the output list
        output.append(y)
    return np.array(output)
```

This code first defines a sigmoid function, which is used in the logistic regression model to map any real-valued number into the range [0, 1]. Then, it defines the logistic regression model, which takes as input the features of a data point and a set of weights, and outputs a probability. Finally, it defines the predict function, which takes as input a dataframe, initializes the weights, iterates over the rows of the dataframe, extracts the features of each row, computes the prediction using the logistic regression model, and appends the prediction to the output list. The function returns the output list as a numpy array.