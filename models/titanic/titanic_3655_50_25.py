import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that if the passenger is a female, in first class, and embarked from Cherbourg, 
        # they have a high probability of survival. 
        # If the passenger is a male, in third class, and embarked from Southampton, 
        # they have a low probability of survival. 
        # The age, number of siblings/spouses, parents/children, and fare are also considered in the prediction.
        # The younger the passenger and the fewer siblings/spouses or parents/children they have, 
        # the higher the probability of survival. 
        # The higher the fare, the higher the probability of survival.

        y = 0.5
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['class_First'] == 1.0:
            y += 0.1
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.1
        if row['age'] <= 30.0:
            y += 0.05
        if row['sibsp'] == 0.0 and row['parch'] == 0.0:
            y += 0.05
        if row['fare'] >= 50.0:
            y += 0.05
        if row['sex_male'] == 1.0:
            y -= 0.3
        if row['class_Third'] == 1.0:
            y -= 0.1
        if row['embark_town_Southampton'] == 1.0:
            y -= 0.1
        if row['age'] > 60.0:
            y -= 0.05
        if row['sibsp'] > 0.0 or row['parch'] > 0.0:
            y -= 0.05
        if row['fare'] < 10.0:
            y -= 0.05

        # Limit the probability between 0 and 1
        y = max(min(y, 1), 0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)