Here is a simple Python code that uses a linear regression model to predict the probability of the target being 1. This code assumes that the input 'x' is a pandas DataFrame with columns 'a', 'b', 'c', 'd', and 'target'.

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

    # Predict the target value for each row
    for index, row in df.iterrows():
        y = weights.dot(row[['a', 'b', 'c', 'd']])
        y = 1 / (1 + np.exp(-y))  # Apply the logistic function to get a probability
        output.append(y)

    return np.array(output)
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. This ensures that all features have the same scale, which is important for the linear regression model.

Next, it calculates the weights for the linear regression model using the formula for the least squares solution: `weights = (X^T * X)^-1 * X^T * y`, where `X` is the matrix of feature values and `y` is the vector of target values.

Finally, it predicts the target value for each row by taking the dot product of the weights and the feature values. It applies the logistic function to this value to get a probability between 0 and 1. The logistic function is commonly used in binary classification problems to convert a linear regression output into a probability.