import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We assume that if the passenger is female, embarked from Cherbourg, and is in first class, 
        # then the probability of survival is high (0.9). 
        # If the passenger is male, embarked from Southampton, and is in third class, 
        # then the probability of survival is low (0.1). 
        # For all other cases, we assume a neutral probability of survival (0.5).
        if row['sex_female'] == 1.0 and row['embarked_C'] == 1.0 and row['class_First'] == 1.0:
            y = 0.9
        elif row['sex_male'] == 1.0 and row['embarked_S'] == 1.0 and row['class_Third'] == 1.0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)