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
        y = weights[0] * row['a'] + weights[1] * row['b'] + weights[2] * row['c'] + weights[3] * row['d']
        output.append(y)

    # Convert the output to probabilities using the sigmoid function
    output = 1 / (1 + np.exp(-np.array(output)))

    return output
```

This code first normalizes the data by subtracting the mean and dividing by the standard deviation of each column. Then it calculates the weights for the linear regression model using the formula for the least squares solution. Finally, it uses these weights to predict the target values and converts these values to probabilities using the sigmoid function.

Please note that this is a very basic model and may not provide accurate predictions for complex data. For more accurate predictions, you may want to consider using a more advanced machine learning model, such as logistic regression or a neural network.