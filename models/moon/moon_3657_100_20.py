Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame with columns named "Feature_1", "Feature_2", and "target".

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = pd.read_csv('data.csv', names=['Feature_1', 'Feature_2', 'target'])

# Split the data into features and target
X = data[['Feature_1', 'Feature_2']]
y = data['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Extract the features for this row
        features = row[['Feature_1', 'Feature_2']].values.reshape(1, -1)
        
        # Use the model to predict the probability of the target being 1
        prob = model.predict_proba(features)[0][1]
        
        output.append(prob)
    return np.array(output)
```

This code first reads the data from a CSV file and splits it into features (X) and target (y). It then trains a logistic regression model on this data. The `predict` function takes a DataFrame of unknown data, extracts the features for each row, and uses the model to predict the probability of the target being 1. These probabilities are collected into an array which is returned by the function.