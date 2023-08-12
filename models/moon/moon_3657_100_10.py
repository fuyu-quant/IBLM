Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame with columns named 'Feature_1', 'Feature_2', and 'target'.

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
        # Extract the features
        features = row[['Feature_1', 'Feature_2']]
        
        # Predict the probability of the target being 1
        prob = model.predict_proba([features])[0][1]
        
        output.append(prob)
    return np.array(output)
```

This code first trains a logistic regression model on the given data. Then, in the `predict` function, it uses this model to predict the probability that the "target" of each row in the input DataFrame is 1. The `predict_proba` method of the logistic regression model returns a 2D array where the first element of each row is the probability that the target is 0 and the second element is the probability that the target is 1. Therefore, we extract the second element to get the required probability.