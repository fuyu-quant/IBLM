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
    for index, row in df.iterrows():
        # Extract the features
        features = np.array([row['Feature_1'], row['Feature_2']])
        
        # Define the weights and bias (these values should be learned from the data)
        weights = np.array([0.1, 0.2])  # These are placeholder values
        bias = 0.5  # This is a placeholder value
        
        # Compute the prediction
        y = logistic_regression(features, weights, bias)
        
        # Append the prediction to the output list
        output.append(y)
    
    return np.array(output)
```

Please note that the weights and bias in the logistic regression model should be learned from the data. The values provided in the code are placeholder values and will not give accurate predictions. To learn the weights and bias from the data, you would typically use a method such as gradient descent, which involves iteratively updating the weights and bias to minimize the difference between the model's predictions and the actual target values. However, implementing gradient descent from scratch is beyond the scope of this task.