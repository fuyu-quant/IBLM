import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can see that the target is more likely to be 1 if the passenger is female, embarked from Cherbourg, is alone, and is in first class. 
        # Conversely, the target is more likely to be 0 if the passenger is male, embarked from Southampton, is not alone, and is in third class.
        # We can use these observations to make a simple prediction.

        score = 0

        # Increase score if passenger is female
        if row['sex_female'] == 1.0:
            score += 1

        # Increase score if passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            score += 1

        # Increase score if passenger is alone
        if row['alone_True'] == 1.0:
            score += 1

        # Increase score if passenger is in first class
        if row['class_First'] == 1.0:
            score += 1

        # Decrease score if passenger is male
        if row['sex_male'] == 1.0:
            score -= 1

        # Decrease score if passenger embarked from Southampton
        if row['embark_town_Southampton'] == 1.0:
            score -= 1

        # Decrease score if passenger is not alone
        if row['alone_False'] == 1.0:
            score -= 1

        # Decrease score if passenger is in third class
        if row['class_Third'] == 1.0:
            score -= 1

        # Normalize the score to a probability between 0 and 1
        y = 1 / (1 + np.exp(-score))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)