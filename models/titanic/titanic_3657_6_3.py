Here is a simple example of a prediction function that uses a basic rule-based approach. This function assumes that the target is more likely to be 1 if the passenger is female, is in first class, and embarked from Cherbourg. This is a very simplistic approach and would likely not perform well in a real-world scenario, but it serves to illustrate the basic structure of a prediction function.

```python
import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 1.0
        else:
            y = 0.0
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)
```

This function could be improved by incorporating more features and using a more sophisticated decision-making process. For example, you could use a decision tree or a logistic regression model to make predictions based on multiple features. However, the question specifies not to use an existing machine learning model, so a rule-based approach is used here.