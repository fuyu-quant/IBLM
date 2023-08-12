import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observation of the data.
        # For example, if the passenger is female, in first class, and embarked from Cherbourg, 
        # the probability of survival is high. 
        # On the other hand, if the passenger is male, in third class, and embarked from Southampton, 
        # the probability of survival is low.

        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 0.9
        elif row['sex_male'] == 1.0 and row['class_Third'] == 1.0 and row['embark_town_Southampton'] == 1.0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)