import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, passengers in first class (pclass=1), female passengers (sex_female=1), passengers who embarked at Cherbourg (embarked_C=1), 
        # passengers who are alone (alone_True=1), passengers who are not adult males (adult_male_False=1), passengers who are women (who_woman=1), 
        # passengers in Deck B, C, D, E (deck_B=1, deck_C=1, deck_D=1, deck_E=1) and passengers who embarked at Cherbourg (embark_town_Cherbourg=1) 
        # are more likely to survive. Therefore, these features are given more weightage in the prediction.
        y = row['pclass']*(-0.15) + row['age']*(-0.02) + row['sibsp']*(-0.05) + row['parch']*0.05 + row['fare']*0.002 + row['sex_female']*0.3 + row['sex_male']*(-0.3) + row['embarked_C']*0.1 + row['embarked_Q']*0.05 + row['embarked_S']*(-0.05) + row['alive_no']*(-0.3) + row['alive_yes']*0.3 + row['alone_False']*(-0.1) + row['alone_True']*0.1 + row['adult_male_False']*0.15 + row['adult_male_True']*(-0.15) + row['who_child']*0.1 + row['who_man']*(-0.15) + row['who_woman']*0.15 + row['class_First']*0.15 + row['class_Second']*0.05 + row['class_Third']*(-0.1) + row['deck_A']*0.05 + row['deck_B']*0.1 + row['deck_C']*0.1 + row['deck_D']*0.1 + row['deck_E']*0.1 + row['deck_F']*0.05 + row['deck_G']*0.05 + row['embark_town_Cherbourg']*0.1 + row['embark_town_Queenstown']*0.05 + row['embark_town_Southampton']*(-0.05)

        # The output is then passed through a sigmoid function to convert it into a probability between 0 and 1.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)