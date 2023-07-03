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
    # ... add the rest of your data here ...
]
df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'target'])

# Split the data into features and target
X = df[['a', 'b', 'c', 'd']]
y = df['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

def predict(x):
    df = x.copy()
    output = model.predict_proba(df)[:, 1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). It then trains a logistic regression model on this data. The `predict` function takes a DataFrame of new data, makes a copy of it, and uses the trained model to predict the probability that the "target" is 1 for each row in the new data. The output is an array of these probabilities.