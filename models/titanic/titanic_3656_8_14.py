import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, it seems that the target is more likely to be 1 if the passenger is female, 
        # embarked from Cherbourg or Southampton, is alone, is not an adult male, is a child or woman, 
        # is in first or second class, and is on deck A, B, C, D, E, or F. 
        # The fare also seems to have a positive correlation with the target.
        # We will use these features to make our prediction.

        y = 0.0
        y += row['sex_female']
        y += row['embarked_C']
        y += row['embarked_S']
        y += row['alone_True']
        y += row['adult_male_False']
        y += row['who_child']
        y += row['who_woman']
        y += row['class_First']
        y += row['class_Second']
        y += row['deck_A']
        y += row['deck_B']
        y += row['deck_C']
        y += row['deck_D']
        y += row['deck_E']
        y += row['deck_F']
        y += row['fare'] / 100.0

        # Normalize the prediction to be between 0 and 1
        y = max(0.0, min(y / 15.0, 1.0))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)