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

    # Predict the target value for each row
    for index, row in df.iterrows():
        y = weights[0]*row['a'] + weights[1]*row['b'] + weights[2]*row['c'] + weights[3]*row['d']
        output.append(y)

    # Convert the output to a probability between 0 and 1
    output = 1 / (1 + np.exp(-np.array(output)))

    return output
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. It then calculates the weights for the linear regression model using the formula (X^T * X)^-1 * X^T * y, where X is the matrix of input features and y is the vector of target values. The code then uses these weights to predict the target value for each row in the DataFrame. Finally, it converts these predicted values to probabilities between 0 and 1 using the logistic function.