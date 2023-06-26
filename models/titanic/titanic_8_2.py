import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        prob = 0
        prob += row['pclass'] * 0.1
        prob += row['age'] * 0.01
        prob += row['sibsp'] * 0.05
        prob += row['parch'] * 0.05
        prob += row['fare'] * 0.001
        prob += row['sex_female'] * 0.3
        prob += row['sex_male'] * -0.3
        prob += row['embarked_C'] * 0.1
        prob += row['embarked_Q'] * 0.05
        prob += row['embarked_S'] * -0.05
        prob += row['alive_no'] * -0.2
        prob += row['alive_yes'] * 0.2
        prob += row['alone_False'] * 0.1
        prob += row['alone_True'] * -0.1
        prob += row['adult_male_False'] * 0.2
        prob += row['adult_male_True'] * -0.2
        prob += row['who_child'] * 0.2
        prob += row['who_man'] * -0.1
        prob += row['who_woman'] * 0.1
        prob += row['class_First'] * 0.2
        prob += row['class_Second'] * 0.1
        prob += row['class_Third'] * -0.1
        prob += row['deck_A'] * 0.05
        prob += row['deck_B'] * 0.1
        prob += row['deck_C'] * 0.15
        prob += row['deck_D'] * 0.2
        prob += row['deck_E'] * 0.25
        prob += row['deck_F'] * 0.3
        prob += row['deck_G'] * 0.35
        prob += row['embark_town_Cherbourg'] * 0.1
        prob += row['embark_town_Queenstown'] * 0.05
        prob += row['embark_town_Southampton'] * -0.05

        # Normalize the probability to be between 0 and 1
        y = 1 / (1 + np.exp(-prob))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)