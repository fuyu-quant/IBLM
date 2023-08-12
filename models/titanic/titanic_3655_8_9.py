import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based system to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if the passenger is female (sex_female=1), the probability of survival (target=1) is high.
        # Similarly, if the passenger is in the first class (pclass=1), the probability of survival is high.
        # We also consider the age of the passenger. Younger passengers have a higher chance of survival.
        # The fare paid by the passenger is also considered. Higher fare indicates higher class and hence higher survival probability.
        # These rules are not perfect and there can be exceptions. But they provide a reasonable starting point for the prediction.

        y = 0.5  # base probability

        if row['sex_female'] == 1:
            y += 0.3
        if row['pclass'] == 1:
            y += 0.2
        if row['age'] <= 30:
            y += 0.1
        if row['fare'] > 20:
            y += 0.1

        # ensure the probability is within [0, 1]
        y = max(0, min(y, 1))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)