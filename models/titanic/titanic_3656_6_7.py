import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based system to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if the passenger is female, in first class, and embarked from Cherbourg, the probability of survival is high.
        # On the other hand, if the passenger is male, in third class, and embarked from Southampton, the probability of survival is low.
        # These rules are not perfect and may not generalize well to unseen data.

        p = 0.5  # base probability

        if row['sex_female'] == 1.0:
            p += 0.3
        if row['pclass'] == 1.0:
            p += 0.2
        if row['embarked_C'] == 1.0:
            p += 0.1

        if row['sex_male'] == 1.0:
            p -= 0.3
        if row['pclass'] == 3.0:
            p -= 0.2
        if row['embarked_S'] == 1.0:
            p -= 0.1

        # ensure probability is within [0, 1]
        p = max(0, min(p, 1))

        y = p

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)