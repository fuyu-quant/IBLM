import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, it seems that the target is more likely to be 1 if the passenger is female, embarked from Cherbourg, is alone, and is in first class.
        # Conversely, the target is more likely to be 0 if the passenger is male, embarked from Southampton, is not alone, and is in third class.
        # We will use these features to make our prediction.

        score = 0

        # Increase score if passenger is female
        if row['sex_female'] == 1.0:
            score += 0.3

        # Increase score if passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            score += 0.2

        # Increase score if passenger is alone
        if row['alone_True'] == 1.0:
            score += 0.1

        # Increase score if passenger is in first class
        if row['class_First'] == 1.0:
            score += 0.4

        # Decrease score if passenger is male
        if row['sex_male'] == 1.0:
            score -= 0.3

        # Decrease score if passenger embarked from Southampton
        if row['embark_town_Southampton'] == 1.0:
            score -= 0.2

        # Decrease score if passenger is not alone
        if row['alone_False'] == 1.0:
            score -= 0.1

        # Decrease score if passenger is in third class
        if row['class_Third'] == 1.0:
            score -= 0.4

        # Normalize score to range between 0 and 1
        y = max(0, min(1, score))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)