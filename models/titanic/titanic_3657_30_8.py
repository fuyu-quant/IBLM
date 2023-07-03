import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, if the passenger is female, if the passenger is in first class, if the passenger is a child, 
        # if the passenger embarked from Cherbourg, if the passenger is alone, and if the passenger's deck is B, D, or E, 
        # then the passenger is more likely to survive. 
        # On the other hand, if the passenger is male, if the passenger is in third class, if the passenger is an adult male, 
        # if the passenger embarked from Southampton, and if the passenger's deck is A, C, F, or G, 
        # then the passenger is less likely to survive.

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['who_child']
        y += row['embark_town_Cherbourg']
        y += row['alone_True']
        y += row['deck_B']
        y += row['deck_D']
        y += row['deck_E']

        y -= row['sex_male']
        y -= row['class_Third']
        y -= row['who_man']
        y -= row['embark_town_Southampton']
        y -= row['deck_A']
        y -= row['deck_C']
        y -= row['deck_F']
        y -= row['deck_G']

        # Normalize the prediction to be between 0 and 1
        y = (y + 8) / 16

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)