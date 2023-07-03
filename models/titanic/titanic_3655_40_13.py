import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can see that the target is more likely to be 1 when:
        # - pclass is lower (1st class has higher survival rate)
        # - age is lower (children have higher survival rate)
        # - fare is higher (people who paid more have higher survival rate)
        # - sex is female (women have higher survival rate)
        # - embarked from Cherbourg (people who embarked from Cherbourg have higher survival rate)
        # - alone is False (people who were not alone have higher survival rate)
        # - adult_male is False (non-adult males have higher survival rate)
        # - who is not man (women and children have higher survival rate)
        # - class is First (1st class passengers have higher survival rate)
        # - deck is not G (people from other decks have higher survival rate)
        # - embark_town is Cherbourg (people who embarked from Cherbourg have higher survival rate)

        y = 0.1 * (3 - row['pclass']) + 0.1 * (1 if row['age'] <= 16 else 0) + 0.1 * (row['fare'] / 100) + \
            0.1 * row['sex_female'] + 0.1 * row['embarked_C'] + 0.1 * (1 - row['alone_True']) + \
            0.1 * (1 - row['adult_male_True']) + 0.1 * (1 if row['who_man'] == 0 else 0) + \
            0.1 * row['class_First'] + 0.1 * (1 if row['deck_G'] == 0 else 0) + 0.1 * row['embark_town_Cherbourg']

        # Normalize the prediction to be between 0 and 1
        y = max(0, min(y, 1))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)