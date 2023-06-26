import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        prob = 0
        if row['pclass'] == 1:
            prob += 0.3
        elif row['pclass'] == 2:
            prob += 0.2
        else:
            prob += 0.1

        if row['sex_female']:
            prob += 0.3
        else:
            prob -= 0.1

        if row['age'] <= 10:
            prob += 0.2
        elif row['age'] <= 30:
            prob += 0.1
        else:
            prob -= 0.1

        if row['fare'] > 50:
            prob += 0.2
        elif row['fare'] > 20:
            prob += 0.1
        else:
            prob -= 0.1

        if row['embarked_C']:
            prob += 0.1
        elif row['embarked_Q']:
            prob += 0.05
        else:
            prob -= 0.05

        if row['sibsp'] > 0 or row['parch'] > 0:
            prob += 0.1
        else:
            prob -= 0.1

        # Normalize the probability to be between 0 and 1
        y = max(min(prob, 1), 0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)