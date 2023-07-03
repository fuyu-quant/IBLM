Here is a simple Python code that uses a basic linear regression model to predict the probability of the target being 1. This code does not use any existing machine learning model, but rather implements the logic of linear regression from scratch.

```python
import numpy as np
import pandas as pd

# Define the data
data = {
    'Feature_1': [0.059, 0.203, -0.016, -0.858, 1.634, -0.52, 1.909, 1.05, 0.121, 0.91],
    'Feature_2': [0.2, 0.944, 0.375, 0.617, -0.302, 0.975, 0.038, 0.149, 0.163, 0.419],
    'target': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
}
df = pd.DataFrame(data)

# Calculate the coefficients of the linear regression model
X = df[['Feature_1', 'Feature_2']].values
y = df['target'].values
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# Define the prediction function
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Calculate the predicted probability
        x = np.array([1, row['Feature_1'], row['Feature_2']])
        y = 1 / (1 + np.exp(-x @ beta))
        output.append(y)
    return np.array(output)
```

This code first calculates the coefficients of the linear regression model using the given data. Then, it defines a prediction function that calculates the predicted probability for each row in the input DataFrame. The predicted probability is calculated using the logistic function, which is commonly used in logistic regression to map any real-valued number into the range [0, 1].