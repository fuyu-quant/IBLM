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
    w = np.array([0.1, 0.1])
    b = 0.1
    for index, row in df.iterrows():
        # Extract the features
        features = np.array([row['Feature_1'], row['Feature_2']])
        # Compute the probability using the logistic regression model
        y = logistic_regression(features, w, b)
        output.append(y)
    return np.array(output)
```

This code first defines the sigmoid function, which is used in the logistic regression model to map any real-valued number into the range [0, 1]. Then it defines the logistic regression model, which takes as input the features of a data point, the weights of the model, and the bias of the model, and outputs the probability that the "target" of the data point is 1. Finally, it defines the predict function, which takes as input a DataFrame, initializes the weights and bias of the model, and computes the probability for each data point in the DataFrame using the logistic regression model.

Please note that this is a very basic implementation of the logistic regression model, and it does not include any training process to learn the optimal weights and bias from the data. Therefore, the accuracy of the predictions may not be very high. To improve the accuracy, you would need to implement a training process, such as gradient descent, to learn the optimal weights and bias from the data.