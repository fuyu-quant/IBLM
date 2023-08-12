import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based system to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if the passenger is female (sex_female=1.0), embarked from Cherbourg (embarked_C=1.0), and travelled in first class (class_First=1.0), 
        # then the probability of survival (target=1) is high.
        # Similarly, if the passenger is male (sex_male=1.0), embarked from Southampton (embark_town_Southampton=1.0), and travelled in third class (class_Third=1.0), 
        # then the probability of survival (target=1) is low.
        # This is a very simplistic approach and may not give accurate results for all cases.

        if row['sex_female'] == 1.0 and row['embarked_C'] == 1.0 and row['class_First'] == 1.0:
            y = 0.9
        elif row['sex_male'] == 1.0 and row['embark_town_Southampton'] == 1.0 and row['class_Third'] == 1.0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)