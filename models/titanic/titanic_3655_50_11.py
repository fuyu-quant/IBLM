import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that if the passenger is a female (sex_female=1), in first class (class_First=1), and embarked from Cherbourg (embark_town_Cherbourg=1), 
        # the probability of survival is high. Conversely, if the passenger is a male (sex_male=1), in third class (class_Third=1), and embarked from Southampton (embark_town_Southampton=1), 
        # the probability of survival is low. This is a simple rule-based approach and does not take into account all the features in the dataset.

        if row['sex_female'] == 1 and row['class_First'] == 1 and row['embark_town_Cherbourg'] == 1:
            y = 0.9
        elif row['sex_male'] == 1 and row['class_Third'] == 1 and row['embark_town_Southampton'] == 1:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)