import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        prob = 0

        # Higher class passengers have a higher probability of survival
        if row['pclass'] == 1:
            prob += 0.6
        elif row['pclass'] == 2:
            prob += 0.4
        else:
            prob += 0.2

        # Female passengers have a higher probability of survival
        if row['sex_female']:
            prob += 0.35

        # Passengers with family members on board have a higher probability of survival
        if row['sibsp'] > 0 or row['parch'] > 0:
            prob += 0.1

        # Passengers who paid a higher fare have a higher probability of survival
        if row['fare'] > df['fare'].median():
            prob += 0.1

        # Passengers who embarked at Cherbourg have a higher probability of survival
        if row['embark_town_Cherbourg']:
            prob += 0.05

        # Normalize the probability to be between 0 and 1
        prob = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)