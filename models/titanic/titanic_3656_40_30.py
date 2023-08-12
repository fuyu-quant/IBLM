import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, passengers in first class (pclass=1), female passengers (sex_female=1), passengers who embarked at Cherbourg (embarked_C=1), 
        # passengers who are alone (alone_True=1), passengers who are children (who_child=1), passengers who are women (who_woman=1), 
        # passengers in Deck B, C, D, E (deck_B=1, deck_C=1, deck_D=1, deck_E=1) are more likely to survive.
        # On the other hand, passengers in third class (pclass=3), male passengers (sex_male=1), passengers who embarked at Southampton (embarked_S=1), 
        # passengers who are not alone (alone_False=1), passengers who are men (who_man=1), passengers in Deck A, F, G (deck_A=1, deck_F=1, deck_G=1) 
        # are less likely to survive.
        # The fare is also considered, assuming that passengers who paid a higher fare are more likely to survive.
        # The age is considered inversely, assuming that younger passengers are more likely to survive.
        # The sibsp (number of siblings/spouses aboard) and parch (number of parents/children aboard) are not considered as they do not seem to have a clear impact on survival.

        y = 0.1*row['pclass'] + 0.2*row['sex_female'] + 0.1*row['embarked_C'] + 0.1*row['alone_True'] + 0.1*row['who_child'] + 0.1*row['who_woman'] + 0.05*row['deck_B'] + 0.05*row['deck_C'] + 0.05*row['deck_D'] + 0.05*row['deck_E'] + 0.05*row['fare'] - 0.05*row['age'] - 0.1*row['pclass'] - 0.2*row['sex_male'] - 0.1*row['embarked_S'] - 0.1*row['alone_False'] - 0.1*row['who_man'] - 0.05*row['deck_A'] - 0.05*row['deck_F'] - 0.05*row['deck_G']

        # The result is then normalized to be between 0 and 1 using the sigmoid function.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)