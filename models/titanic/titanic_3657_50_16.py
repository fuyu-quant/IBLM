import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, it seems that the survival rate is higher for females, people in first class, and those who embarked from Cherbourg.
        # Therefore, we will assign higher probability values to these groups.
        # This is a simple heuristic and does not take into account interactions between variables or non-linear relationships.

        prob = 0.5  # base probability

        # Increase probability if passenger is female
        if row['sex_female'] == 1.0:
            prob += 0.3

        # Increase probability if passenger is in first class
        if row['class_First'] == 1.0:
            prob += 0.1

        # Increase probability if passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            prob += 0.1

        # Ensure probability is within [0, 1]
        prob = min(max(prob, 0), 1)

        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)