Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code does not use any existing machine learning model, but rather implements the logistic regression model from scratch.

```python
import numpy as np
import pandas as pd

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the logistic regression model
def logistic_regression(X, y, num_steps, learning_rate):
    # Initialize weights
    weights = np.zeros(X.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(X, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = y - predictions
        gradient = np.dot(X.T, output_error_signal)
        weights += learning_rate * gradient
        
    return weights

# Define the prediction function
def predict(x):
    df = x.copy()
    output = []
    
    # Separate features and target
    X = df[['Feature_1', 'Feature_2']].values
    y = df['target'].values
    
    # Train the logistic regression model
    weights = logistic_regression(X, y, num_steps = 200000, learning_rate = 0.01)
    
    for index, row in df.iterrows():
        # Calculate score
        score = np.dot(weights, [row['Feature_1'], row['Feature_2']])
        
        # Calculate probability
        probability = sigmoid(score)
        
        output.append(probability)
        
    return np.array(output)
```

This code first defines the sigmoid function, which is used to map any real-valued number into the range between 0 and 1. Then it defines the logistic regression model, which uses gradient descent to find the weights that minimize the error between the predicted and actual target values. Finally, it defines the prediction function, which uses the trained logistic regression model to predict the probability that the "target" of the unknown data is 1.