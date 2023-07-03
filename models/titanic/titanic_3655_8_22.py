import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observation of the given data.
        # For example, if 'sex_female' is 1, the probability of survival seems to be higher.
        # Similarly, if 'pclass' is lower, the survival rate seems to be higher.
        # These rules are not perfect and may not generalize well to unseen data.
        # For a more robust solution, a machine learning model should be trained on the data.

        y = 0.5  # base probability

        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.2
        if row['pclass'] == 2.0:
            y += 0.1
        if row['fare'] > 20.0:
            y += 0.1
        if row['age'] < 18.0:
            y += 0.1

        # ensure the probability is within [0, 1]
        y = min(max(y, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)