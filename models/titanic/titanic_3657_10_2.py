import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for target=1 if the person is female, embarked from Cherbourg, is alone, is an adult male, belongs to first class, and has a deck. 
        # Similarly, lower probability is given if the person is male, embarked from Queenstown or Southampton, is not alone, is not an adult male, belongs to second or third class, and does not have a deck.

        y = 0.5  # start with a neutral probability

        # adjust probability based on gender
        if row['sex_female'] == 1.0:
            y += 0.1
        elif row['sex_male'] == 1.0:
            y -= 0.1

        # adjust probability based on embarkment
        if row['embarked_C'] == 1.0:
            y += 0.1
        elif row['embarked_Q'] == 1.0 or row['embarked_S'] == 1.0:
            y -= 0.1

        # adjust probability based on alone status
        if row['alone_True'] == 1.0:
            y += 0.1
        elif row['alone_False'] == 1.0:
            y -= 0.1

        # adjust probability based on adult male status
        if row['adult_male_True'] == 1.0:
            y += 0.1
        elif row['adult_male_False'] == 1.0:
            y -= 0.1

        # adjust probability based on class
        if row['class_First'] == 1.0:
            y += 0.1
        elif row['class_Second'] == 1.0 or row['class_Third'] == 1.0:
            y -= 0.1

        # adjust probability based on deck
        if row['deck_A'] == 1.0 or row['deck_B'] == 1.0 or row['deck_C'] == 1.0 or row['deck_D'] == 1.0 or row['deck_E'] == 1.0 or row['deck_F'] == 1.0 or row['deck_G'] == 1.0:
            y += 0.1

        # ensure probability stays within [0,1]
        y = max(0, min(y, 1))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)