Here is a simple Python code that uses a linear regression model to predict the probability of the target being 1. This code assumes that the input `x` is a pandas DataFrame with columns 'a', 'b', 'c', 'd', and 'target'.

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
    df = (df - means) / stds

    # Calculate the weights for the linear regression model
    weights = np.linalg.inv(df[['a', 'b', 'c', 'd']].T.dot(df[['a', 'b', 'c', 'd']])).dot(df[['a', 'b', 'c', 'd']].T).dot(df['target'])

    for index, row in df.iterrows():
        # Calculate the dot product of the weights and the input values
        y = weights.dot(row[['a', 'b', 'c', 'd']])

        # Apply the sigmoid function to get the probability
        y = 1 / (1 + np.exp(-y))

        output.append(y)

    return np.array(output)
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation. Then it calculates the weights for the linear regression model using the formula `(X^T * X)^-1 * X^T * y`, where `X` is the input data and `y` is the target data. For each row in the input data, it calculates the dot product of the weights and the input values, and applies the sigmoid function to get the probability of the target being 1. The sigmoid function is used to map the output of the linear regression model to the range [0, 1], which can be interpreted as a probability.