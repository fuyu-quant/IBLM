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
        weights = np.array([0.5, 0.5, 0.5])
        
        # Compute the prediction
        y = logistic_regression(features, weights)
        
        # Append the prediction to the output
        output.append(y)
    return np.array(output)
```

This code first defines the sigmoid function, which is used in the logistic regression model to map any real-valued number into the range [0, 1]. Then, it defines the logistic regression model, which takes as input the features of a data point and a set of weights, and outputs a probability between 0 and 1.

The predict function takes as input a DataFrame, makes a copy of it, and then iterates over its rows. For each row, it extracts the features, defines a set of weights (in this case, all weights are set to 0.5, but in a real-world scenario these weights should be learned from the data), computes the prediction using the logistic regression model, and appends the prediction to the output. Finally, it returns the output as a NumPy array.