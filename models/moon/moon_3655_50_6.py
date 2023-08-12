Here is a simple Python code that uses a linear regression model to predict the probability of the target being 1. This code assumes that the input `x` is a pandas DataFrame with the columns `Feature_1`, `Feature_2`, and `target`.

```python
import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []

    # Calculate the mean and standard deviation of each feature for each target value
    means = df.groupby('target').mean()
    stds = df.groupby('target').std()

    # Calculate the probability of each target value
    prob_target = df['target'].value_counts(normalize=True)

    for index, row in df.iterrows():
        # Calculate the probability of the data point under each target value
        prob_0 = np.exp(-((row['Feature_1'] - means.loc[0, 'Feature_1'])**2 / (2 * stds.loc[0, 'Feature_1']**2) + 
                          (row['Feature_2'] - means.loc[0, 'Feature_2'])**2 / (2 * stds.loc[0, 'Feature_2']**2))) * prob_target[0]
        prob_1 = np.exp(-((row['Feature_1'] - means.loc[1, 'Feature_1'])**2 / (2 * stds.loc[1, 'Feature_1']**2) + 
                          (row['Feature_2'] - means.loc[1, 'Feature_2'])**2 / (2 * stds.loc[1, 'Feature_2']**2))) * prob_target[1]

        # Normalize the probabilities so they sum to 1
        prob_0, prob_1 = prob_0 / (prob_0 + prob_1), prob_1 / (prob_0 + prob_1)

        # Append the probability of the target being 1 to the output
        output.append(prob_1)

    return np.array(output)
```

This code uses a Gaussian naive Bayes classifier to predict the probability of the target being 1. It assumes that the features are normally distributed and that they are independent given the target value. The code first calculates the mean and standard deviation of each feature for each target value, as well as the overall probability of each target value. It then calculates the probability of each data point under each target value, normalizes these probabilities so they sum to 1, and appends the probability of the target being 1 to the output.