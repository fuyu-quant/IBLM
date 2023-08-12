Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the data is stored in a pandas DataFrame.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = [
    [1.131,-0.562,1.0], [1.314,-0.443,1.0], [2.05,0.283,1.0], [0.175,-0.015,1.0], [1.21,-0.488,1.0],
    # ... add all the other data points here ...
    [-1.039,0.041,0.0]
]
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2', 'target'])

# Split the data into features (X) and target (y)
X = df[['Feature_1', 'Feature_2']]
y = df['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define the prediction function
def predict(x):
    df = x.copy()
    output = model.predict_proba(df)[:, 1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). Then, it trains a logistic regression model on this data. The `predict` function takes a DataFrame of unknown data, makes a copy of it, and uses the trained model to predict the probability that the "target" is 1 for each row in the DataFrame. The probabilities are returned as a numpy array.