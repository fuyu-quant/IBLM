import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if 'sex_female' is 1, then the probability of target being 1 is high.
        # Similarly, if 'alive_yes' is 1, then the probability of target being 1 is high.
        # We are also considering 'age' and 'fare' in our prediction.
        # If 'age' is less and 'fare' is high, then the probability of target being 1 is high.
        # These rules are not perfect and may not work well on unseen data.
        # For a more accurate prediction, a machine learning model should be trained on the data.

        y = 0.5  # base probability

        if row['sex_female'] == 1.0:
            y += 0.3
        if row['alive_yes'] == 1.0:
            y += 0.2
        if row['age'] < 30.0 and row['fare'] > 20.0:
            y += 0.1

        # limit the probability between 0 and 1
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)