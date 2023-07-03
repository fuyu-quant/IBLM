import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if 'pclass' is 1.0 or 'sex_female' is 1.0, the probability of target being 1 is high.
        # Similarly, if 'age' is less than 18.0 or 'fare' is high, the probability of target being 1 is also high.
        # These rules are not perfect and may not work well on unseen data.

        p = 0.5  # base probability

        if row['pclass'] == 1.0:
            p += 0.3
        if row['sex_female'] == 1.0:
            p += 0.2
        if row['age'] < 18.0:
            p += 0.1
        if row['fare'] > 50.0:
            p += 0.1

        # ensure the probability is within [0, 1]
        p = min(max(p, 0), 1)

        y = p

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)