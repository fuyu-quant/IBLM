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
    w = np.array([0.1, 0.2])
    b = 0.1
    for index, row in df.iterrows():
        # Extract the features
        features = row[['Feature_1', 'Feature_2']].values
        # Predict the probability using the logistic regression model
        y = logistic_regression(features, w, b)
        output.append(y)
    return np.array(output)
```

This code first defines the sigmoid function, which is used in the logistic regression model to map any real-valued number into the range [0, 1]. Then it defines the logistic regression model, which takes as input the features of a data point, the weights of the model, and a bias term, and outputs a probability between 0 and 1.

The predict function takes as input a DataFrame containing the features of the data points to be predicted. It initializes the weights and bias of the logistic regression model, and then iterates over the rows of the DataFrame. For each row, it extracts the features, predicts the probability using the logistic regression model, and appends the predicted probability to the output list. Finally, it returns the output list as a numpy array.