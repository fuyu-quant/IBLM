import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We are assuming that if the passenger is a female (sex_female=1.0), 
        # or if the passenger is a child (who_child=1.0), or if the passenger is in first class (class_First=1.0),
        # then the passenger has a high probability of survival (target=1).
        # Otherwise, the passenger has a low probability of survival.
        # This is based on the historical fact that women, children, and first-class passengers were given priority during the evacuation of the Titanic.
        if row['sex_female'] == 1.0 or row['who_child'] == 1.0 or row['class_First'] == 1.0:
            y = 0.9
        else:
            y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)