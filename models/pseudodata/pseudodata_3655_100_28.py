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

    # Predict the target for each row
    for index, row in df.iterrows():
        y = weights.dot(row[['a', 'b', 'c', 'd']])
        output.append(y)

    # Convert the output to probabilities using the sigmoid function
    output = 1 / (1 + np.exp(-np.array(output)))

    return output
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. Then it calculates the weights for the linear regression model using the formula `weights = (X^T * X)^-1 * X^T * y`, where `X` is the matrix of input features and `y` is the target vector. Finally, it predicts the target for each row by taking the dot product of the weights and the input features, and converts these predictions to probabilities using the sigmoid function.