import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, if the passenger is female, if the passenger is a child, if the passenger is in first class, 
        # if the passenger embarked from Cherbourg, if the passenger is alone, and if the passenger's deck is B, D, or E, 
        # these are all factors that increase the likelihood of survival.
        # On the other hand, if the passenger is male, if the passenger is an adult male, if the passenger is in third class, 
        # if the passenger embarked from Southampton, and if the passenger's deck is A, C, F, or G, 
        # these are all factors that decrease the likelihood of survival.
        # The weights for these features are determined based on their relative importance in determining survival.

        y = 0.2*row['sex_female'] + 0.15*row['who_child'] + 0.15*row['class_First'] + 0.1*row['embarked_C'] + 0.1*row['alone_True'] + 0.1*(row['deck_B'] or row['deck_D'] or row['deck_E']) - 0.2*row['sex_male'] - 0.15*row['adult_male_True'] - 0.15*row['class_Third'] - 0.1*row['embarked_S'] - 0.1*(row['deck_A'] or row['deck_C'] or row['deck_F'] or row['deck_G'])

        # Normalize the prediction to be between 0 and 1
        y = (y + 1) / 2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)