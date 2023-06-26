import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given conditions
        prob = 0
        if row['pclass'] == 2 and row['sex_female'] and row['embarked_S'] and row['alive_yes']:
            prob += 0.6
        if row['age'] <= 30 and row['sibsp'] == 0 and row['parch'] == 0:
            prob += 0.3
        if row['fare'] <= 30:
            prob += 0.1

        # Normalize the probability to be between 0 and 1
        y = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)