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
        # Calculate the linear regression prediction
        y = weights[0]*row['a'] + weights[1]*row['b'] + weights[2]*row['c'] + weights[3]*row['d']

        # Convert the prediction to a probability using the logistic function
        y = 1 / (1 + np.exp(-y))

        output.append(y)

    return np.array(output)
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. It then calculates the weights for the linear regression model using the formula for the least squares solution. For each row in the DataFrame, it calculates the linear regression prediction and converts this to a probability using the logistic function. The probabilities are returned as a numpy array.