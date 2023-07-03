import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple heuristic to predict the target.
        # We are assuming that the target is more likely to be 1 if the passenger is female, is in first class, and is alone.
        # This is based on the historical fact that in the Titanic disaster, women, children, and first-class passengers were given priority for lifeboats.
        # We are also assuming that the target is more likely to be 1 if the passenger embarked from Cherbourg, based on the data provided.
        # This is a very simplistic model and would likely not perform well on a larger, more diverse dataset.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.3
        if row['alone_True'] == 1.0:
            y += 0.2
        if row['embarked_C'] == 1.0:
            y += 0.2

        # We normalize the prediction to the range [0, 1].
        y = min(max(y, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)