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
    weights = np.linalg.inv(df.T.dot(df)).dot(df.T).dot(df['target'])

    # Predict the target values
    for index, row in df.iterrows():
        y = weights[0]*row['a'] + weights[1]*row['b'] + weights[2]*row['c'] + weights[3]*row['d']
        y = 1 / (1 + np.exp(-y))  # Apply the logistic function to get a probability
        output.append(y)

    return np.array(output)
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. This ensures that all features have the same scale and that the weights of the linear regression model are not biased towards features with larger values.

The weights of the linear regression model are then calculated using the formula for the least squares solution: `weights = (X^T * X)^-1 * X^T * y`, where `X` is the matrix of feature values and `y` is the vector of target values.

Finally, the code predicts the target values by multiplying the feature values of each row by the corresponding weights and summing the results. The logistic function is applied to these values to convert them into probabilities between 0 and 1.