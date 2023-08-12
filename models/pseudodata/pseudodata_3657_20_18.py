Here is a simple Python code that uses a logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = [[0.71,-1.08,-0.83,-1.21,1.0],
[-0.25,0.49,0.32,0.51,0.0],
[0.22,-0.0,-0.16,-0.12,1.0],
[-0.64,1.27,0.83,1.31,0.0],
[2.04,2.13,-0.86,0.47,1.0],
[-1.41,-1.27,0.65,-0.18,0.0],
[0.3,-1.15,-0.55,-1.03,1.0],
[-2.7,1.35,2.36,2.52,0.0],
[0.42,0.22,-0.24,-0.07,1.0],
[-0.62,1.21,0.81,1.26,0.0],
[0.98,1.09,-0.39,0.28,1.0],
[-0.76,-0.97,0.27,-0.31,0.0],
[0.16,0.2,-0.06,0.06,1.0],
[-1.1,1.05,1.11,1.4,0.0],
[1.55,-1.17,-1.47,-1.74,1.0],
[-0.67,-0.59,0.31,-0.07,0.0],
[0.03,-0.23,-0.09,-0.19,1.0],
[-0.72,-0.69,0.32,-0.12,0.0],
[1.39,-0.34,-1.11,-1.03,1.0],
[-0.72,0.63,0.71,0.87,0.0]]

df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'target'])

# Split the data into features and target
X = df[['a', 'b', 'c', 'd']]
y = df['target']

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

def predict(x):
    df = x.copy()
    output = model.predict_proba(df)[:,1]
    return output
```

This code first prepares the data and splits it into features (X) and target (y). Then it trains a logistic regression model on this data. The predict function takes a DataFrame of new data, makes a copy of it, and uses the trained model to predict the probability that the target is 1 for each row in the new data. The output is an array of these probabilities.