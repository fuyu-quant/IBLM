import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        prob = 0
        prob += 0.3 if row['pclass'] == 1 else 0
        prob += 0.2 if row['age'] <= 18 else 0
        prob += 0.1 if row['sibsp'] == 0 else 0
        prob += 0.1 if row['parch'] == 0 else 0
        prob += 0.1 if row['fare'] > 50 else 0
        prob += 0.2 if row['sex_female'] else 0
        prob += 0.1 if row['embarked_C'] else 0
        prob += 0.1 if row['alone_True'] else 0
        prob += 0.1 if row['who_child'] else 0
        prob += 0.1 if row['class_First'] else 0
        prob += 0.1 if row['deck_B'] or row['deck_C'] or row['deck_D'] or row['deck_E'] else 0
        prob += 0.1 if row['embark_town_Cherbourg'] else 0

        # Normalize the probability to be between 0 and 1
        y = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)