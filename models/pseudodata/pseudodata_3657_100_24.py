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
    df = (df - means) / stds

    # Calculate the weights for the linear regression model
    weights = np.linalg.inv(df[['a', 'b', 'c', 'd']].T.dot(df[['a', 'b', 'c', 'd']])).dot(df[['a', 'b', 'c', 'd']].T).dot(df['target'])

    for index, row in df.iterrows():
        # Calculate the linear regression prediction
        y = weights[0]*row['a'] + weights[1]*row['b'] + weights[2]*row['c'] + weights[3]*row['d']

        # Convert the prediction to a probability using the logistic function
        y = 1 / (1 + np.exp(-y))

        output.append(y)

    return np.array(output)
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. It then calculates the weights for the linear regression model using the formula for the least squares solution. For each row in the DataFrame, it calculates a linear regression prediction using these weights, and then converts this prediction to a probability using the logistic function. The output is an array of these probabilities.