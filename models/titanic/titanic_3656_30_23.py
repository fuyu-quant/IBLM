import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can see that the survival rate is higher for females, children, passengers in first class, and those who embarked from Cherbourg.
        # We will assign higher probability values for these conditions.

        prob = 0.5  # start with a base probability of 0.5

        # Increase probability if passenger is female
        if row['sex_female'] == 1.0:
            prob += 0.2

        # Increase probability if passenger is a child
        if row['who_child'] == 1.0:
            prob += 0.1

        # Increase probability if passenger is in first class
        if row['class_First'] == 1.0:
            prob += 0.1

        # Increase probability if passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            prob += 0.1

        # Decrease probability if passenger is male and in third class
        if row['sex_male'] == 1.0 and row['class_Third'] == 1.0:
            prob -= 0.2

        # Ensure probability stays within [0,1]
        prob = max(0, min(prob, 1))

        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)