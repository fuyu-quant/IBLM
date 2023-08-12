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
        # - embarked from Cherbourg (people from Cherbourg have higher survival rate)
        # - alone is False (people with family have higher survival rate)
        # - who is woman or child (women and children have higher survival rate)
        # - class is First (1st class has higher survival rate)
        # - deck is B, D, E (people on these decks have higher survival rate)
        # - embark town is Cherbourg (people from Cherbourg have higher survival rate)

        y = 0
        y += row['pclass'] * -0.1
        y += row['age'] * -0.02
        y += row['fare'] * 0.02
        y += row['sex_female'] * 0.3
        y += row['embarked_C'] * 0.1
        y += row['alone_False'] * 0.1
        y += row['who_child'] * 0.1
        y += row['who_woman'] * 0.1
        y += row['class_First'] * 0.1
        y += row['deck_B'] * 0.1
        y += row['deck_D'] * 0.1
        y += row['deck_E'] * 0.1
        y += row['embark_town_Cherbourg'] * 0.1

        # Normalize the output to be between 0 and 1
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)