Here is a simple Python code that predicts the probability of "target" being 1 based on the given data. This code uses a simple rule-based approach to predict the probability. It checks if the passenger is a female, if the passenger is in first class, and if the passenger is a child. If all these conditions are met, it predicts a high probability of survival (0.9). If none of these conditions are met, it predicts a low probability of survival (0.1). For all other cases, it predicts a medium probability of survival (0.5).

```python
import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        if row['sex_female'] == 1 and row['pclass'] == 1 and row['who_child'] == 1:
            y = 0.9
        elif row['sex_female'] == 0 and row['pclass'] == 3 and row['who_child'] == 0:
            y = 0.1
        else:
            y = 0.5
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)
```

Please note that this is a very simple rule-based approach and may not provide accurate predictions for all cases. For more accurate predictions, you may need to use a machine learning model.