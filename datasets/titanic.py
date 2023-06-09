import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Please describe the process required to make the prediction below.

        # If the passenger is a woman or a child, predict survival (1)
        if row['sex'] == 'female' or row['age'] < 18:
            y = 1
        # If the passenger is a man and in first or second class, predict survival (1)
        elif row['pclass'] in [1, 2] and row['sex'] == 'male':
            y = 1
        # Otherwise, predict non-survival (0)
        else:
            y = 0

        output.append(y)
    return np.array(output)