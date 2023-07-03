import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if 'pclass' is 1 or 2, 'sex_female' is 1, 'fare' is high, 'age' is low, 'sibsp' is low, 'parch' is low, 
        # 'embarked_C' is 1, 'alive_yes' is 1, 'alone_True' is 1, 'adult_male_False' is 1, 'who_woman' is 1, 'class_First' is 1, 
        # 'deck_B' or 'deck_D' or 'deck_E' is 1, 'embark_town_Cherbourg' is 1, then the probability of 'target' being 1 is high.

        y = 0.5  # base probability

        if row['pclass'] <= 2:
            y += 0.1
        if row['sex_female'] == 1:
            y += 0.1
        if row['fare'] > 20:
            y += 0.1
        if row['age'] < 30:
            y += 0.1
        if row['sibsp'] == 0:
            y += 0.1
        if row['parch'] == 0:
            y += 0.1
        if row['embarked_C'] == 1:
            y += 0.1
        if row['alive_yes'] == 1:
            y += 0.1
        if row['alone_True'] == 1:
            y += 0.1
        if row['adult_male_False'] == 1:
            y += 0.1
        if row['who_woman'] == 1:
            y += 0.1
        if row['class_First'] == 1:
            y += 0.1
        if row['deck_B'] == 1 or row['deck_D'] == 1 or row['deck_E'] == 1:
            y += 0.1
        if row['embark_town_Cherbourg'] == 1:
            y += 0.1

        # limit the probability between 0 and 1
        y = max(0, min(y, 1))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)