import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, it seems that the target is more likely to be 1 when:
        # - pclass is lower (1 or 2)
        # - age is lower
        # - fare is higher
        # - sex is female
        # - embarked from Cherbourg or Southampton
        # - alive is yes
        # - alone is False
        # - adult_male is False
        # - who is woman or child
        # - class is First or Second
        # - deck is B, C, D, E
        # - embark_town is Cherbourg or Southampton

        y = 0
        y += 0.1 if row['pclass'] < 3 else 0
        y += 0.1 if row['age'] < 30 else 0
        y += 0.1 if row['fare'] > 30 else 0
        y += 0.1 if row['sex_female'] == 1 else 0
        y += 0.1 if row['embarked_C'] == 1 or row['embarked_S'] == 1 else 0
        y += 0.1 if row['alive_yes'] == 1 else 0
        y += 0.1 if row['alone_False'] == 1 else 0
        y += 0.1 if row['adult_male_False'] == 1 else 0
        y += 0.1 if row['who_woman'] == 1 or row['who_child'] == 1 else 0
        y += 0.1 if row['class_First'] == 1 or row['class_Second'] == 1 else 0
        y += 0.1 if row['deck_B'] == 1 or row['deck_C'] == 1 or row['deck_D'] == 1 or row['deck_E'] == 1 else 0
        y += 0.1 if row['embark_town_Cherbourg'] == 1 or row['embark_town_Southampton'] == 1 else 0

        # Normalize the prediction to be between 0 and 1
        y = min(max(y, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)