import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is that if the passenger is a female, in first class, and embarked from Cherbourg, 
        # they have a high probability of survival. This is based on the historical data from the Titanic disaster.
        if row['sex_female'] == 1 and row['class_First'] == 1 and row['embark_town_Cherbourg'] == 1:
            y = 0.9
        # If the passenger is a male, in third class, and embarked from Southampton, they have a low probability of survival.
        elif row['sex_male'] == 1 and row['class_Third'] == 1 and row['embark_town_Southampton'] == 1:
            y = 0.1
        # For all other passengers, we will assign a moderate probability of survival.
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)