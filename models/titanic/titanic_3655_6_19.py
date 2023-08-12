import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based system to predict the target.
        # The rules are based on the observation of the given data.
        # For example, if 'sex_female' is 1, the probability of survival is high.
        # Similarly, if 'class_First' is 1, the probability of survival is also high.
        # These rules are not perfect and may not work well on unseen data.
        # For a more accurate prediction, a machine learning model should be trained on the data.

        y = 0.5  # base probability

        if row['sex_female'] == 1.0:
            y += 0.3
        if row['class_First'] == 1.0:
            y += 0.2
        if row['fare'] > 20.0:
            y += 0.1
        if row['age'] < 10.0:
            y += 0.1

        # ensure the probability is within [0, 1]
        y = min(max(y, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)