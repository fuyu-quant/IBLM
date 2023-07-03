Here is a simple Python code that uses a basic linear regression model to predict the probability of the target being 1. This code assumes that the input 'x' is a pandas DataFrame with columns 'a', 'b', 'c', 'd', and 'target'.

```python
import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []

    # Calculate the mean and standard deviation of each column
    means = df.mean()
    stds = df.std()

    # Normalize the data
    for column in df.columns:
        df[column] = (df[column] - means[column]) / stds[column]

    # Calculate the weights for the linear regression model
    weights = np.linalg.inv(df[['a', 'b', 'c', 'd']].T.dot(df[['a', 'b', 'c', 'd']])).dot(df[['a', 'b', 'c', 'd']].T).dot(df['target'])

    # Predict the target for each row
    for index, row in df.iterrows():
        y = weights[0]*row['a'] + weights[1]*row['b'] + weights[2]*row['c'] + weights[3]*row['d']
        y = 1 / (1 + np.exp(-y))  # Apply the logistic function to get a probability
        output.append(y)

    return np.array(output)
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. This ensures that all features have the same scale and that the weights of the linear regression model are not biased towards features with larger scales.

The weights of the linear regression model are then calculated using the formula for the least squares solution: `weights = (X^T * X)^-1 * X^T * y`, where `X` is the matrix of feature values and `y` is the vector of target values.

Finally, the code predicts the target for each row by calculating the dot product of the feature values and the weights, and then applying the logistic function to get a probability between 0 and 1. The logistic function is used because it maps any real-valued number to a value between 0 and 1, which makes it suitable for predicting probabilities.