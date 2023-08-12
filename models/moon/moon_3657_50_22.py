Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame with columns named 'Feature_1', 'Feature_2', and 'target'.

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
    [0.033,0.291,1.0],
    [1.257,-0.493,1.0],
    [0.339,-0.323,1.0],
    [1.961,0.033,1.0],
    [1.42,-0.498,1.0],
    [1.918,0.142,1.0],
    [1.847,0.148,1.0],
    [1.807,-0.143,1.0],
    [2.033,0.291,1.0],
    [1.175,-0.457,1.0],
    [1.913,-0.001,1.0],
    [0.013,0.429,1.0],
    [0.248,-0.174,1.0],
    [1.292,-0.426,1.0],
    [0.345,-0.323,1.0],
    [0.023,0.307,1.0],
    [0.887,-0.492,1.0],
    [0.057,0.419,1.0],
    [-0.008,0.208,1.0],
    [1.47,-0.395,1.0],
    [1.397,-0.513,1.0]
]
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2', 'target'])

# Train the model
X = df[['Feature_1', 'Feature_2']]
y = df['target']
model = LogisticRegression()
model.fit(X, y)

# Define the prediction function
def predict(x):
    df = x.copy()
    output = model.predict_proba(df[['Feature_1', 'Feature_2']])[:, 1]
    return output
```

You can use the `predict` function to predict the probability that the "target" of the unknown data is 1. For example:

```python
unknown_data = pd.DataFrame([[1.5, -0.5]], columns=['Feature_1', 'Feature_2'])
print(predict(unknown_data))
```

This will output the predicted probability that the "target" of the unknown data is 1.