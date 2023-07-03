Here is a simple Python code that uses a linear regression model to predict the probability of the target being 1. This code assumes that the input 'x' is a pandas DataFrame with columns 'Feature_1', 'Feature_2', and 'target'.

```python
import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []

    # Calculate the mean and standard deviation of each feature for each target value
    means = df.groupby('target').mean()
    stds = df.groupby('target').std()

    # Calculate the probability of each row belonging to target 1
    for index, row in df.iterrows():
        # Calculate the z-score for each feature
        z1 = (row['Feature_1'] - means.loc[1, 'Feature_1']) / stds.loc[1, 'Feature_1']
        z2 = (row['Feature_2'] - means.loc[1, 'Feature_2']) / stds.loc[1, 'Feature_2']

        # Calculate the probability of the row belonging to target 1
        p1 = np.exp(-0.5 * (z1**2 + z2**2)) / (2 * np.pi * stds.loc[1, 'Feature_1'] * stds.loc[1, 'Feature_2'])

        # Calculate the z-score for each feature
        z1 = (row['Feature_1'] - means.loc[0, 'Feature_1']) / stds.loc[0, 'Feature_1']
        z2 = (row['Feature_2'] - means.loc[0, 'Feature_2']) / stds.loc[0, 'Feature_2']

        # Calculate the probability of the row belonging to target 0
        p0 = np.exp(-0.5 * (z1**2 + z2**2)) / (2 * np.pi * stds.loc[0, 'Feature_1'] * stds.loc[0, 'Feature_2'])

        # The final prediction is the probability of the row belonging to target 1
        y = p1 / (p1 + p0)
        output.append(y)

    return np.array(output)
```

This code uses a Gaussian distribution to model the probability of each feature given the target value. The final prediction is the probability of the row belonging to target 1, calculated using Bayes' theorem.