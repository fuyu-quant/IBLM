import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, it seems that the survival rate is higher for females, people in first class, and those who embarked from Cherbourg.
        # Also, it seems that children and women have a higher survival rate than men.
        # Therefore, we will give higher probability values to these categories.

        y = 0.5  # base probability

        # Increase probability if passenger is female
        if row['sex_female'] == 1.0:
            y += 0.2

        # Increase probability if passenger is in first class
        if row['class_First'] == 1.0:
            y += 0.1

        # Increase probability if passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.05

        # Increase probability if passenger is a child or a woman
        if row['who_child'] == 1.0 or row['who_woman'] == 1.0:
            y += 0.1

        # Decrease probability if passenger is a man
        if row['who_man'] == 1.0:
            y -= 0.1

        # Ensure probability is within [0, 1]
        y = max(0, min(y, 1))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)