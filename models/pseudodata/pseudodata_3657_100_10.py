Here is a simple Python code that uses a linear regression model to predict the probability of the target being 1. This code assumes that the input 'x' is a pandas DataFrame with columns 'a', 'b', 'c', 'd', and 'target'.

```python
import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []

    # Calculate the mean and standard deviation of each column
    mean_a = df['a'].mean()
    std_a = df['a'].std()
    mean_b = df['b'].mean()
    std_b = df['b'].std()
    mean_c = df['c'].mean()
    std_c = df['c'].std()
    mean_d = df['d'].mean()
    std_d = df['d'].std()

    # Normalize the data
    df['a'] = (df['a'] - mean_a) / std_a
    df['b'] = (df['b'] - mean_b) / std_b
    df['c'] = (df['c'] - mean_c) / std_c
    df['d'] = (df['d'] - mean_d) / std_d

    # Calculate the weights for the linear regression model
    weights = np.linalg.inv(df[['a', 'b', 'c', 'd']].T.dot(df[['a', 'b', 'c', 'd']])).dot(df[['a', 'b', 'c', 'd']].T).dot(df['target'])

    for index, row in df.iterrows():
        # Calculate the predicted value
        y = row[['a', 'b', 'c', 'd']].dot(weights)

        # Convert the predicted value to a probability
        y = 1 / (1 + np.exp(-y))

        output.append(y)

    return np.array(output)
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation. Then it calculates the weights for the linear regression model using the formula `weights = (X^T * X)^-1 * X^T * y`, where `X` is the matrix of input features and `y` is the vector of target values. Finally, it calculates the predicted value for each row and converts it to a probability using the logistic function `1 / (1 + exp(-y))`.