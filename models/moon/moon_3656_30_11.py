Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the data is stored in a pandas DataFrame.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = [
    [1.342,-0.412,1.0],
    [2.029,0.302,1.0],
    [0.532,-0.396,1.0],
    [0.021,0.333,1.0],
    [1.731,-0.241,1.0],
    [0.753,-0.613,1.0],
    [1.957,0.304,1.0],
    [1.209,-0.53,1.0],
    [1.689,-0.229,1.0],
    [0.06,0.525,1.0],
    [0.731,-0.436,1.0],
    [0.256,-0.106,1.0],
    [1.516,-0.398,1.0],
    [0.749,-0.513,1.0],
    [0.084,0.316,1.0],
    [1.643,-0.379,1.0],
    [1.28,-0.498,1.0],
    [1.998,0.356,1.0],
    [0.986,-0.413,1.0],
    [0.095,0.099,1.0],
    [1.878,0.06,1.0],
    [0.094,-0.052,1.0],
    [1.56,-0.322,1.0],
    [0.582,-0.449,1.0],
    [0.481,-0.398,1.0],
    [1.991,0.284,1.0],
    [0.091,0.222,1.0],
    [0.985,-0.464,1.0],
    [-0.008,0.469,1.0],
    [2.011,0.462,1.0]
]
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2', 'target'])

# Split the data into features and target
X = df[['Feature_1', 'Feature_2']]
y = df['target']

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

def predict(x):
    df = x.copy()
    output = model.predict_proba(df)[:,1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). Then, it trains a logistic regression model using the features and target. The `predict` function takes a DataFrame of unknown data, makes a copy of it, and uses the trained model to predict the probability that the "target" is 1 for each row in the DataFrame. The function returns these probabilities as a numpy array.