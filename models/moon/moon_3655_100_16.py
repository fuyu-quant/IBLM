Here is a simple Python code that uses a linear regression model to predict the probability of the target being 1. This code assumes that the input 'x' is a pandas DataFrame with columns 'Feature_1', 'Feature_2', and 'target'.

```python
import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []

    # Calculate the mean and standard deviation of Feature_1 and Feature_2 for target 0 and 1
    mean_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].mean()
    std_0 = df[df['target'] == 0][['Feature_1', 'Feature_2']].std()
    mean_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].mean()
    std_1 = df[df['target'] == 1][['Feature_1', 'Feature_2']].std()

    for index, row in df.iterrows():
        # Calculate the probability of the data point belonging to target 0 and 1 using Gaussian distribution
        prob_0 = (1 / (np.sqrt(2 * np.pi * std_0['Feature_1']**2)) * np.exp(-(row['Feature_1'] - mean_0['Feature_1'])**2 / (2 * std_0['Feature_1']**2))) * \
                  (1 / (np.sqrt(2 * np.pi * std_0['Feature_2']**2)) * np.exp(-(row['Feature_2'] - mean_0['Feature_2'])**2 / (2 * std_0['Feature_2']**2)))

        prob_1 = (1 / (np.sqrt(2 * np.pi * std_1['Feature_1']**2)) * np.exp(-(row['Feature_1'] - mean_1['Feature_1'])**2 / (2 * std_1['Feature_1']**2))) * \
                  (1 / (np.sqrt(2 * np.pi * std_1['Feature_2']**2)) * np.exp(-(row['Feature_2'] - mean_1['Feature_2'])**2 / (2 * std_1['Feature_2']**2)))

        # Normalize the probabilities so they sum to 1
        total_prob = prob_0 + prob_1
        prob_0 /= total_prob
        prob_1 /= total_prob

        # Append the probability of the target being 1 to the output
        output.append(prob_1)

    return np.array(output)
```

This code uses a simple Gaussian Naive Bayes classifier to predict the probability of the target being 1. It first calculates the mean and standard deviation of 'Feature_1' and 'Feature_2' for target 0 and 1. Then, for each data point, it calculates the probability of the data point belonging to target 0 and 1 using the Gaussian distribution. Finally, it normalizes the probabilities so they sum to 1 and appends the probability of the target being 1 to the output.