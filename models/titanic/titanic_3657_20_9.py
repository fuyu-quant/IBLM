import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher weightage to the features that are more likely to result in survival (target=1)
        # and lower weightage to the features that are more likely to result in non-survival (target=0).
        # The weights are determined based on the data provided.

        y = 0.0
        y += row['pclass'] * -0.15
        y += row['age'] * -0.02
        y += row['sibsp'] * -0.05
        y += row['parch'] * 0.05
        y += row['fare'] * 0.002
        y += row['sex_female'] * 0.3
        y += row['sex_male'] * -0.3
        y += row['embarked_C'] * 0.1
        y += row['embarked_Q'] * 0.05
        y += row['embarked_S'] * -0.05
        y += row['alive_no'] * -0.5
        y += row['alive_yes'] * 0.5
        y += row['alone_False'] * 0.1
        y += row['alone_True'] * -0.1
        y += row['adult_male_False'] * 0.2
        y += row['adult_male_True'] * -0.2
        y += row['who_child'] * 0.2
        y += row['who_man'] * -0.2
        y += row['who_woman'] * 0.2
        y += row['class_First'] * 0.2
        y += row['class_Second'] * 0.1
        y += row['class_Third'] * -0.1
        y += row['deck_A'] * 0.05
        y += row['deck_B'] * 0.1
        y += row['deck_C'] * 0.1
        y += row['deck_D'] * 0.1
        y += row['deck_E'] * 0.1
        y += row['deck_F'] * 0.05
        y += row['deck_G'] * 0.05
        y += row['embark_town_Cherbourg'] * 0.1
        y += row['embark_town_Queenstown'] * 0.05
        y += row['embark_town_Southampton'] * -0.05

        # Convert the final score to a probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)