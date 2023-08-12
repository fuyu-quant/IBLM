import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority during the evacuation.
        # The port of embarkation is also considered as passengers from Cherbourg had a higher survival rate.
        # The age of the passenger is also considered, giving higher survival probability to children.
        # The fare paid by the passenger is also considered, assuming that passengers who paid higher fares might have been given priority during the evacuation.
        # The number of siblings/spouses and parents/children aboard is also considered, assuming that passengers with family members might have helped each other to survive.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.2
        if row['class_First'] == 1.0:
            y += 0.2
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.1
        if row['age'] <= 18.0:
            y += 0.1
        if row['fare'] >= 50.0:
            y += 0.1
        if row['sibsp'] > 0.0 or row['parch'] > 0.0:
            y += 0.1
        if y > 1.0:
            y = 1.0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)