import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can see that the target is more likely to be 1 when:
        # - pclass is lower (1st class passengers have higher survival rate)
        # - age is lower (children have higher survival rate)
        # - sex is female (women have higher survival rate)
        # - fare is higher (passengers who paid more have higher survival rate)
        # - embarked from Cherbourg (passengers from Cherbourg have higher survival rate)
        # - travelling alone (passengers travelling alone have higher survival rate)
        # - deck is B, D, or E (passengers on these decks have higher survival rate)

        y = 0
        y += 0.2 if row['pclass'] == 1 else 0
        y += 0.2 if row['age'] <= 18 else 0
        y += 0.2 if row['sex_female'] == 1 else 0
        y += 0.2 if row['fare'] > 30 else 0
        y += 0.2 if row['embarked_C'] == 1 else 0
        y += 0.2 if row['alone_True'] == 1 else 0
        y += 0.2 if row['deck_B'] == 1 or row['deck_D'] == 1 or row['deck_E'] == 1 else 0

        # Normalize the prediction to be between 0 and 1
        y = min(max(y, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)