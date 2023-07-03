import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, passengers in first class (pclass), female passengers (sex_female), passengers who embarked at Cherbourg (embarked_C), 
        # passengers who are alone (alone_True), passengers who are children (who_child), passengers in deck B, C, D, E (deck_B, deck_C, deck_D, deck_E) 
        # and passengers who embarked at Cherbourg (embark_town_Cherbourg) are more likely to survive.
        # On the other hand, passengers in third class (pclass), male passengers (sex_male), passengers who embarked at Southampton (embarked_S), 
        # passengers who are not alone (alone_False), passengers who are man (who_man), passengers in deck A, F, G (deck_A, deck_F, deck_G) 
        # and passengers who embarked at Southampton (embark_town_Southampton) are less likely to survive.
        # The fare is also considered, assuming that passengers who paid a higher fare are more likely to survive.

        y = 0.1*row['pclass'] + 0.3*row['sex_female'] - 0.3*row['sex_male'] + 0.1*row['embarked_C'] - 0.1*row['embarked_S'] + 0.1*row['alone_True'] - 0.1*row['alone_False'] + 0.3*row['who_child'] - 0.3*row['who_man'] + 0.1*row['deck_B'] + 0.1*row['deck_C'] + 0.1*row['deck_D'] + 0.1*row['deck_E'] - 0.1*row['deck_A'] - 0.1*row['deck_F'] - 0.1*row['deck_G'] + 0.1*row['embark_town_Cherbourg'] - 0.1*row['embark_town_Southampton'] + 0.01*row['fare']
        y = 1 / (1 + np.exp(-y))  # Apply sigmoid function to convert y to a probability

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)