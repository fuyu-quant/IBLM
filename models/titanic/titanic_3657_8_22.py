Here is a simple Python code that uses a basic logistic regression model to predict the probability that the "target" of the unknown data is 1. This code assumes that the input data is a pandas DataFrame.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Prepare the data
data = [
    [3.0,18.0,0.0,1.0,9.35,1.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1],
    [3.0,31.0,0.0,0.0,7.775,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0]
    # Add more data here...
]
df = pd.DataFrame(data, columns=['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'alive_no', 'alive_yes', 'alone_False', 'alone_True', 'adult_male_False', 'adult_male_True', 'who_child', 'who_man', 'who_woman', 'class_First', 'class_Second', 'class_Third', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'embark_town_Cherbourg', 'embark_town_Queenstown', 'embark_town_Southampton', 'target'])

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

def predict(x):
    df = x.copy()
    output = model.predict_proba(df)[:, 1]
    return output
```

This code first prepares the data and splits it into features and target. Then it trains a logistic regression model using the data. The `predict` function takes a DataFrame of unknown data, makes a copy of it, and uses the trained model to predict the probability that the "target" is 1 for each row in the DataFrame. The function returns these probabilities as a numpy array.