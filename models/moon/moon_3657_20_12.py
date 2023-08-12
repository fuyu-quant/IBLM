Here is a simple Python code that uses a basic logistic regression model to predict the probability that the "target" of the unknown data is 1. This code does not use any existing machine learning model, but rather implements the logistic regression model from scratch.

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
        gradient = np.dot(X.T, (predictions - y))
        weights -= learning_rate * gradient

    return weights

# Define the prediction function
def predict(x):
    df = x.copy()
    output = []

    # Extract features and target
    X = df[['Feature_1', 'Feature_2']].values
    y = df['target'].values

    # Train the logistic regression model
    weights = logistic_regression(X, y, num_steps=5000, learning_rate=0.01)

    for index, row in df.iterrows():
        # Calculate the score
        score = np.dot(row[['Feature_1', 'Feature_2']].values, weights)
        
        # Calculate the probability
        probability = sigmoid(score)
        
        output.append(probability)

    return np.array(output)
```

This code first defines the sigmoid function, which is used to map any real-valued number into the range [0, 1], which can be interpreted as probabilities. Then it defines the logistic regression model, which is trained using gradient descent. The prediction function first extracts the features and target from the input DataFrame, trains the logistic regression model, and then uses the trained model to predict the probabilities for each row in the DataFrame.