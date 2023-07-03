Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame with columns named 'Feature_1', 'Feature_2', and 'target'.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = [
    [0.059,0.2,1.0],
    [0.203,0.944,0.0],
    [-0.016,0.375,1.0],
    [-0.858,0.617,0.0],
    [1.634,-0.302,1.0],
    [-0.52,0.975,0.0],
    [1.909,0.038,1.0],
    [1.05,0.149,0.0],
    [0.121,0.163,1.0],
    [0.91,0.419,0.0],
    [0.39,-0.299,1.0],
    [-0.229,0.959,0.0],
    [0.226,-0.153,1.0],
    [-0.632,0.769,0.0],
    [0.075,0.075,1.0],
    [-1.045,0.185,0.0],
    [0.38,-0.31,1.0],
    [-0.895,0.425,0.0],
    [1.87,0.074,1.0],
    [0.461,0.849,0.0],
    [1.01,-0.535,1.0],
    [-0.26,0.915,0.0],
    [1.989,0.135,1.0],
    [-0.854,0.397,0.0],
    [1.976,0.191,1.0],
    [0.065,0.977,0.0],
    [0.109,0.109,1.0],
    [-0.884,0.339,0.0],
    [0.28,-0.288,1.0],
    [0.004,0.972,0.0],
    [2.007,0.231,1.0],
    [-0.772,0.586,0.0],
    [0.395,-0.351,1.0],
    [0.759,0.736,0.0],
    [0.207,-0.137,1.0],
    [0.18,1.003,0.0],
    [0.748,-0.418,1.0],
    [0.725,0.651,0.0],
    [0.426,-0.32,1.0],
    [-0.877,0.431,0.0],
    [1.995,0.393,1.0],
    [-0.715,0.661,0.0],
    [0.83,-0.516,1.0],
    [-0.819,0.587,0.0],
    [0.02,0.366,1.0],
    [-0.533,0.785,0.0],
    [2.029,0.2,1.0],
    [-1.142,0.108,0.0],
    [1.655,-0.233,1.0],
    [0.924,-0.07,0.0]
]
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2', 'target'])

# Split the data into features and target
X = df[['Feature_1', 'Feature_2']]
y = df['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define the prediction function
def predict(x):
    df = x.copy()
    output = model.predict_proba(df[['Feature_1', 'Feature_2']])[:, 1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). It then trains a logistic regression model on this data. The `predict` function takes a DataFrame as input, makes a copy of it, and uses the trained model to predict the probability that the "target" is 1 for each row in the DataFrame. The output is an array of probabilities.