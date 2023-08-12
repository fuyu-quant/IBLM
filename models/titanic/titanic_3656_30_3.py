import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that if the passenger is a female, in first class, and embarked from Cherbourg, 
        # they have a high probability of survival. Conversely, if the passenger is a male, in third class, 
        # and embarked from Southampton, they have a low probability of survival. 
        # This is a simple heuristic based on the data and may not be 100% accurate.

        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 0.9
        elif row['sex_male'] == 1.0 and row['class_Third'] == 1.0 and row['embark_town_Southampton'] == 1.0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)