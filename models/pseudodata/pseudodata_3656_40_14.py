Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = [
    [0.65,0.66,2.23,1.39,1.0],
    [0.05,-0.43,-1.06,-1.02,0.0],
    [-1.72,0.17,-1.02,0.84,1.0],
    [-0.76,-0.32,-1.47,-0.57,0.0],
    [-1.69,0.75,0.48,2.2,1.0],
    [-0.15,-0.56,-1.56,-1.28,0.0],
    [1.24,0.18,1.52,0.12,1.0],
    [1.14,0.07,1.16,-0.12,0.0],
    [-0.42,0.62,1.22,1.57,1.0],
    [2.33,0.0,1.98,-0.59,0.0],
    [-2.22,0.12,-1.57,0.86,1.0],
    [-0.79,-0.49,-1.94,-0.97,0.0],
    [-1.09,0.59,0.6,1.69,1.0],
    [0.22,-0.37,-0.76,-0.93,0.0],
    [-1.98,-0.23,-2.27,-0.04,1.0],
    [2.34,-0.95,-0.45,-2.84,0.0],
    [-1.86,0.52,-0.25,1.7,1.0],
    [0.14,-0.7,-1.67,-1.69,0.0],
    [-1.56,0.31,-0.51,1.14,1.0],
    [3.75,-0.97,0.7,-3.25,0.0],
    [-1.47,0.59,0.27,1.77,1.0],
    [0.71,-0.39,-0.41,-1.12,0.0],
    [-0.42,0.63,1.25,1.59,1.0],
    [2.27,-0.37,0.98,-1.45,0.0],
    [-2.13,-0.12,-2.12,0.25,1.0],
    [1.06,-0.3,0.14,-0.98,0.0],
    [-1.76,0.12,-1.18,0.73,1.0],
    [1.41,-0.11,0.92,-0.61,0.0],
    [1.28,0.25,1.74,0.27,1.0],
    [2.71,-0.59,0.79,-2.08,0.0],
    [-1.68,0.24,-0.8,1.01,1.0],
    [2.06,-0.76,-0.21,-2.34,0.0],
    [3.44,-0.08,2.7,-1.08,1.0],
    [2.65,-0.75,0.33,-2.45,0.0],
    [-2.12,-0.23,-2.39,-0.01,1.0],
    [-0.07,-0.35,-0.97,-0.82,0.0],
    [0.78,0.34,1.53,0.6,1.0],
    [3.52,-0.74,1.08,-2.66,0.0],
    [-1.65,0.79,0.64,2.3,1.0],
    [-0.17,-0.45,-1.29,-1.01,0.0]
]
df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'target'])

# Split the data into features and target
X = df[['a', 'b', 'c', 'd']]
y = df['target']

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Define the prediction function
def predict(x):
    df = x.copy()
    probabilities = model.predict_proba(df)[:, 1]
    return probabilities
```

This code first prepares the data and splits it into features (X) and target (y). It then trains a logistic regression model on this data. The `predict` function takes a DataFrame of unknown data, makes a copy of it, and uses the trained model to predict the probabilities that the "target" is 1. The function returns these probabilities as a numpy array.