import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if 'pclass' is 1 or 2, 'sex_female' is 1, 'embarked_C' is 1, 'alive_yes' is 1, 'alone_False' is 1, 'adult_male_False' is 1, 'who_woman' is 1, 'class_First' or 'class_Second' is 1, 'deck_B' or 'deck_D' or 'deck_E' is 1, 'embark_town_Cherbourg' is 1, then the probability of target being 1 is high.
        # Otherwise, the probability of target being 1 is low.

        if (row['pclass'] <= 2) and (row['sex_female'] == 1) and (row['embarked_C'] == 1) and (row['alive_yes'] == 1) and (row['alone_False'] == 1) and (row['adult_male_False'] == 1) and (row['who_woman'] == 1) and ((row['class_First'] == 1) or (row['class_Second'] == 1)) and ((row['deck_B'] == 1) or (row['deck_D'] == 1) or (row['deck_E'] == 1)) and (row['embark_town_Cherbourg'] == 1):
            y = 1
        else:
            y = 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)