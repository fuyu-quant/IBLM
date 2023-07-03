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
        # - sex is female (women have higher survival rate)
        # - fare is higher (people who paid more have higher survival rate)
        # - embarked from Cherbourg (people who embarked from Cherbourg have higher survival rate)
        # - alone is False (people who were not alone have higher survival rate)
        # - adult_male is False (non-adult males have higher survival rate)
        # - who is not man (women and children have higher survival rate)
        # - class is not Third (1st and 2nd class have higher survival rate)
        # - deck is not unknown (people with known deck have higher survival rate)
        # - embark_town is not Southampton (people who embarked from Cherbourg and Queenstown have higher survival rate)

        y = 0.5  # base probability

        # adjust probability based on the conditions
        if row['pclass'] == 1:
            y += 0.1
        if row['age'] <= 18:
            y += 0.1
        if row['sex_female'] == 1:
            y += 0.1
        if row['fare'] > df['fare'].median():
            y += 0.1
        if row['embarked_C'] == 1:
            y += 0.1
        if row['alone_False'] == 1:
            y += 0.1
        if row['adult_male_False'] == 1:
            y += 0.1
        if row['who_man'] == 0:
            y += 0.1
        if row['class_Third'] == 0:
            y += 0.1
        if row['deck_A':'deck_G'].sum() > 0:
            y += 0.1
        if row['embark_town_Southampton'] == 0:
            y += 0.1

        # limit the probability between 0 and 1
        y = max(0, min(y, 1))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)