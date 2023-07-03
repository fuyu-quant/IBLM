import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, it seems that the survival rate is higher for females, people in first class, and those who embarked from Cherbourg.
        # We will assign higher probability values for these conditions.
        # This is a simple heuristic and does not take into account interactions between variables or more complex patterns in the data.
        # For a more accurate model, machine learning techniques should be used.

        p = 0.5  # base probability

        # Increase probability if passenger is female
        if row['sex_female'] == 1.0:
            p += 0.2

        # Increase probability if passenger is in first class
        if row['class_First'] == 1.0:
            p += 0.1

        # Increase probability if passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            p += 0.1

        # Ensure probability is within [0, 1]
        p = min(max(p, 0), 1)

        y = p

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)