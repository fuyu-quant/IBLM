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
    # ... add the rest of your data here ...
    [0.924,-0.514,1.0]
]
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2', 'target'])

# Split the data into features and target
X = df[['Feature_1', 'Feature_2']]
y = df['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

def predict(x):
    df = x.copy()
    output = model.predict_proba(df)[:, 1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). It then trains a logistic regression model on this data. The `predict` function takes a DataFrame of unknown data, makes a copy of it, and uses the trained model to predict the probability that the "target" is 1 for each row in the DataFrame. The output is an array of these probabilities.