import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        y = 0
        if row['pclass'] == 1:
            y += 0.3
        if row['sex_female'] == 1:
            y += 0.3
        if row['fare'] > 50:
            y += 0.2
        if row['age'] < 10:
            y += 0.2
        if row['sibsp'] == 0 and row['parch'] == 0:
            y -= 0.1
        if row['embarked_C'] == 1:
            y += 0.1
        if row['class_First'] == 1:
            y += 0.1
        if row['deck_A'] == 1 or row['deck_B'] == 1 or row['deck_C'] == 1 or row['deck_D'] == 1 or row['deck_E'] == 1:
            y += 0.1
        if y > 1:
            y = 1
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)