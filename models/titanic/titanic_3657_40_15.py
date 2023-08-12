import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can see that the survival rate is higher for females, children, first class passengers, and those who embarked from Cherbourg.
        # We can also see that the survival rate is lower for males, adults, third class passengers, and those who embarked from Southampton.
        # Therefore, we can use these factors to predict the survival rate.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.6
        if row['who_child'] == 1.0:
            y += 0.6
        if row['class_First'] == 1.0:
            y += 0.6
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.6

        if row['sex_male'] == 1.0:
            y -= 0.6
        if row['who_man'] == 1.0:
            y -= 0.6
        if row['class_Third'] == 1.0:
            y -= 0.6
        if row['embark_town_Southampton'] == 1.0:
            y -= 0.6

        # Normalize the prediction to be between 0 and 1
        y = (y + 2.4) / 4.8

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)