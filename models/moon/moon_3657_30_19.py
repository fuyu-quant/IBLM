Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = [
    [1.131,-0.562,1.0],
    [1.314,-0.443,1.0],
    [2.05,0.283,1.0],
    [0.175,-0.015,1.0],
    [1.21,-0.488,1.0],
    [-0.061,0.273,1.0],
    [0.218,-0.014,1.0],
    [0.889,-0.538,1.0],
    [0.068,0.073,1.0],
    [1.132,-0.492,1.0],
    [0.607,-0.348,1.0],
    [-0.008,0.258,1.0],
    [1.83,-0.08,1.0],
    [1.888,0.053,1.0],
    [-0.1,0.346,1.0],
    [0.624,-0.48,1.0],
    [1.892,0.063,1.0],
    [0.307,-0.272,1.0],
    [0.882,-0.494,1.0],
    [1.403,-0.352,1.0],
    [0.745,-0.509,1.0],
    [0.379,-0.362,1.0],
    [1.485,-0.409,1.0],
    [0.945,-0.436,1.0],
    [1.231,-0.455,1.0],
    [0.061,0.154,1.0],
    [2.065,0.465,1.0],
    [1.613,-0.293,1.0],
    [0.134,-0.079,1.0],
    [0.033,0.291,1.0]
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
    output = model.predict_proba(df)[:, 1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). Then it trains a logistic regression model on this data. The `predict` function takes a DataFrame of unknown data, makes a copy of it, and uses the trained model to predict the probability that the "target" is 1 for each row in the DataFrame. The output is an array of these probabilities.