Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame with columns named 'Feature_1', 'Feature_2', and 'target'.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = [
    [1.342,-0.412,1.0],
    [2.029,0.302,1.0],
    # ... add the rest of your data here ...
    [1.466,-0.368,1.0]
]
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2', 'target'])

# Train a logistic regression model
model = LogisticRegression()
model.fit(df[['Feature_1', 'Feature_2']], df['target'])

def predict(x):
    df = x.copy()
    output = model.predict_proba(df[['Feature_1', 'Feature_2']])[:, 1]
    return output
```

This code first prepares the data and trains a logistic regression model on it. The `predict` function then uses this model to predict the probability that the "target" of the unknown data is 1. The `predict_proba` method of the model returns a 2D array where the second column represents the probability that the target is 1.