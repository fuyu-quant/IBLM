import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can see that the survival rate is higher for females, children, passengers in first class, and those who embarked from Cherbourg.
        # We can also see that the survival rate is lower for males, adults, passengers in third class, and those who embarked from Southampton.
        # We will use these observations to make our predictions.

        y = 0.5  # start with a base probability of 0.5

        # increase probability for females
        if row['sex_female'] == 1.0:
            y += 0.3

        # decrease probability for males
        if row['sex_male'] == 1.0:
            y -= 0.3

        # increase probability for children
        if row['who_child'] == 1.0:
            y += 0.2

        # decrease probability for adults
        if row['who_man'] == 1.0 or row['who_woman'] == 1.0:
            y -= 0.2

        # increase probability for first class passengers
        if row['class_First'] == 1.0:
            y += 0.2

        # decrease probability for third class passengers
        if row['class_Third'] == 1.0:
            y -= 0.2

        # increase probability for passengers who embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.1

        # decrease probability for passengers who embarked from Southampton
        if row['embark_town_Southampton'] == 1.0:
            y -= 0.1

        # ensure probability stays within [0, 1]
        y = max(0, min(y, 1))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)