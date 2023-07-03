import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observation of the given data.
        # For example, if the passenger is female, embarked from Cherbourg, and is in first class, 
        # the probability of survival is high. Similarly, if the passenger is male, embarked from Southampton, 
        # and is in third class, the probability of survival is low.

        if row['sex_female'] == 1 and row['embarked_C'] == 1 and row['class_First'] == 1:
            y = 0.9
        elif row['sex_male'] == 1 and row['embarked_S'] == 1 and row['class_Third'] == 1:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)