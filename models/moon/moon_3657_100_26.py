Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame with columns named 'Feature_1', 'Feature_2', and 'target'.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = pd.DataFrame({
    'Feature_1': [1.131, 1.314, 2.05, 0.175, 1.21, -0.061, 0.218, 0.889, 0.068, 1.132, 0.607, -0.008, 1.83, 1.888, -0.1, 0.624, 1.892, 0.307, 0.882, 1.403, 0.745, 0.379, 1.485, 0.945, 1.231, 0.061, 2.065, 1.613, 0.134, 0.033, 1.257, 0.339, 1.961, 1.42, 1.918, 1.847, 1.807, 2.033, 1.175, 1.913, 0.013, 0.248, 1.292, 0.345, 0.023, 0.887, 0.057, -0.008, 1.47, 1.397, 2.053, 0.562, 0.799, 0.631, 1.795, 0.886, 0.083, 1.815, 0.085, 1.339, 0.554, 0.748, 1.589, 0.162, 1.49, 0.457, 0.974, 0.621, 0.489, 1.576, 0.962, 1.03, 1.733, 1.101, 0.564, 1.446, 0.043, 0.091, 1.802, 1.974, 1.861, 0.875, 0.415, 1.0, 1.647, 1.666, 1.006, 0.19, 0.026, 0.339, 0.011, 1.614, 0.042, 0.596, 0.176, 0.226, 0.181, 2.011, 1.226, 0.924],
    'Feature_2': [-0.562, -0.443, 0.283, -0.015, -0.488, 0.273, -0.014, -0.538, 0.073, -0.492, -0.348, 0.258, -0.08, 0.053, 0.346, -0.48, 0.063, -0.272, -0.494, -0.352, -0.509, -0.362, -0.409, -0.436, -0.455, 0.154, 0.465, -0.293, -0.079, 0.291, -0.493, -0.323, 0.033, -0.498, 0.142, 0.148, -0.143, 0.291, -0.457, -0.001, 0.429, -0.174, -0.426, -0.323, 0.307, -0.492, 0.419, 0.208, -0.395, -0.513, 0.281, -0.496, -0.595, -0.524, 0.139, -0.462, -0.116, -0.078, 0.203, -0.482, -0.36, -0.473, -0.345, -0.085, -0.44, -0.309, -0.532, -0.415, -0.33, -0.44, -0.467, -0.502, -0.049, -0.497, -0.396, -0.352, 0.017, 0.163, -0.192, 0.36, -0.06, -0.462, -0.292, -0.441, -0.191, -0.204, -0.49, 0.06, 0.424, -0.317, 0.472, -0.272, 0.246, -0.412, -0.06, -0.142, -0.257, 0.36, -0.487, -0.514],
    'target': [1.0]*100
})

# Split the data into features and target
X = data[['Feature_1', 'Feature_2']]
y = data['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

def predict(x):
    df = x.copy()
    output = model.predict_proba(df)[:, 1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). It then trains a logistic regression model on this data. The `predict` function takes a DataFrame of unknown data and uses the trained model to predict the probability that the 'target' is 1 for each row. The probabilities are returned as a numpy array.