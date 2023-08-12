import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can see that the target is more likely to be 1 when the following conditions are met:
        # - pclass is 1 or 2
        # - age is less than 30
        # - fare is greater than 20
        # - sex_female is 1
        # - embarked_C is 1
        # - alive_yes is 1
        # - alone_True is 1
        # - adult_male_False is 1
        # - who_woman is 1
        # - class_First is 1
        # - deck_B, deck_C, deck_D, deck_E are 1
        # - embark_town_Cherbourg is 1

        # We will assign a higher probability if more of these conditions are met.
        conditions_met = sum([row['pclass'] < 3, row['age'] < 30, row['fare'] > 20, row['sex_female'] == 1, row['embarked_C'] == 1, row['alive_yes'] == 1, row['alone_True'] == 1, row['adult_male_False'] == 1, row['who_woman'] == 1, row['class_First'] == 1, row['deck_B'] == 1, row['deck_C'] == 1, row['deck_D'] == 1, row['deck_E'] == 1, row['embark_town_Cherbourg'] == 1])

        # The probability is the number of conditions met divided by the total number of conditions.
        y = conditions_met / 15

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)