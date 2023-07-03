Here is a simple Python code that uses a linear regression model to predict the probability of the target being 1. This code assumes that the input 'x' is a pandas DataFrame with columns 'Feature_1', 'Feature_2', and 'target'.

```python
import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []

    # Calculate the mean and standard deviation of each feature for target 1 and 0
    mean_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].mean()
    std_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].std()
    mean_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].mean()
    std_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].std()

    for index, row in df.iterrows():
        # Calculate the probability of the data point belonging to target 1 and 0 using Gaussian distribution
        prob_1 = (1 / (np.sqrt(2 * np.pi * std_1['Feature_1']**2))) * np.exp(-((row['Feature_1'] - mean_1['Feature_1'])**2 / (2 * std_1['Feature_1']**2))) * \
                  (1 / (np.sqrt(2 * np.pi * std_1['Feature_2']**2))) * np.exp(-((row['Feature_2'] - mean_1['Feature_2'])**2 / (2 * std_1['Feature_2']**2)))

        prob_0 = (1 / (np.sqrt(2 * np.pi * std_0['Feature_1']**2))) * np.exp(-((row['Feature_1'] - mean_0['Feature_1'])**2 / (2 * std_0['Feature_1']**2))) * \
                  (1 / (np.sqrt(2 * np.pi * std_0['Feature_2']**2))) * np.exp(-((row['Feature_2'] - mean_0['Feature_2'])**2 / (2 * std_0['Feature_2']**2)))

        # Calculate the probability of the target being 1
        y = prob_1 / (prob_1 + prob_0)
        output.append(y)

    return np.array(output)
```

This code calculates the probability of each data point belonging to target 1 and 0 using the Gaussian distribution, and then calculates the probability of the target being 1 by dividing the probability of target 1 by the sum of the probabilities of target 1 and 0.