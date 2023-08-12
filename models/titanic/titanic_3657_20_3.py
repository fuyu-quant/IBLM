import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if the passenger is female (sex_female=1), embarked from Cherbourg (embarked_C=1), and is in first class (class_First=1), 
        # then the probability of survival is high (close to 1). 
        # On the other hand, if the passenger is male (sex_male=1), embarked from Southampton (embark_town_Southampton=1), and is in third class (class_Third=1), 
        # then the probability of survival is low (close to 0).
        # These rules are not perfect and there will be exceptions, but they should give a reasonable prediction based on the given data.

        if row['sex_female'] == 1 and row['embarked_C'] == 1 and row['class_First'] == 1:
            y = 0.9
        elif row['sex_male'] == 1 and row['embark_town_Southampton'] == 1 and row['class_Third'] == 1:
            y = 0.1
        else:
            y = 0.5  # if none of the above conditions are met, we assign a neutral probability

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)