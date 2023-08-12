Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame with columns named 'Feature_1', 'Feature_2', and 'target'.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = pd.read_csv('data.csv')  # assuming the data is in a CSV file
X = data[['Feature_1', 'Feature_2']]
y = data['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        proba = model.predict_proba(row.values.reshape(1, -1))
        y = proba[0][1]
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)
```

This code first trains a logistic regression model on the given data. Then, in the `predict` function, it uses the trained model to predict the probability that the "target" of each row in the input DataFrame is 1. The predicted probabilities are returned as a numpy array.