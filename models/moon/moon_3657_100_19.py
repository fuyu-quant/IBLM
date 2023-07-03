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

# Train a logistic regression model
model = LogisticRegression()
model.fit(df[['Feature_1', 'Feature_2']], df['target'])

def predict(x):
    df = x.copy()
    output = model.predict_proba(df[['Feature_1', 'Feature_2']])[:, 1]
    return output
```

This code first trains a logistic regression model on the provided data. The `predict` function then uses this model to predict the probability that the "target" of the unknown data is 1. The `predict_proba` method of the logistic regression model returns a 2D array where the first column is the probability that the target is 0 and the second column is the probability that the target is 1. We only return the second column as the output.