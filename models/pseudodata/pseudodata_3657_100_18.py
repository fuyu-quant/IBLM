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

    # Predict the target value for each row
    for index, row in df.iterrows():
        y = weights[0]*row['a'] + weights[1]*row['b'] + weights[2]*row['c'] + weights[3]*row['d']
        y = 1 / (1 + np.exp(-y))  # Apply the logistic function to get a probability between 0 and 1
        output.append(y)

    return np.array(output)
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. Then it calculates the weights for the linear regression model using the formula `weights = (X^T * X)^-1 * X^T * y`, where `X` is the matrix of input features and `y` is the target vector. Finally, it predicts the target value for each row by taking the dot product of the weights and the input features, and applies the logistic function to get a probability between 0 and 1.