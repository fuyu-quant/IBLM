Here is a simple Python code that uses a linear regression model to predict the probability of the target being 1. This code assumes that the input 'x' is a pandas DataFrame with columns 'Feature_1' and 'Feature_2'.

```python
import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []

    # Calculate the mean and standard deviation of the features for target 0 and 1
    mean_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].mean()
    std_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].std()
    mean_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].mean()
    std_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].std()

    for index, row in df.iterrows():
        # Calculate the probability of the data point under the distribution of target 0 and 1
        prob_0 = np.exp(-((row[['Feature_1', 'Feature_2']] - mean_0) ** 2 / (2 * std_0 ** 2)).sum())
        prob_1 = np.exp(-((row[['Feature_1', 'Feature_2']] - mean_1) ** 2 / (2 * std_1 ** 2)).sum())

        # The predicted probability of target 1 is the probability of the data point under the distribution of target 1
        # divided by the sum of the probabilities under the distributions of target 0 and 1
        y = prob_1 / (prob_0 + prob_1)
        output.append(y)

    return np.array(output)
```

This code uses a Gaussian distribution to model the features for each target value. The probability of a data point given a target value is calculated as the product of the probabilities of its features under the Gaussian distribution of the corresponding target value. The predicted probability of the target being 1 is then calculated as the ratio of the probability of the data point under the distribution of target 1 to the sum of the probabilities under the distributions of target 0 and 1.