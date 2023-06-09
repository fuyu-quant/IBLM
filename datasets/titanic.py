import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can observe that women and children have a higher survival rate.
        # Also, passengers with higher class (First and Second) have a higher survival rate.
        # We will use these observations to predict the target.

        if row['who'] == 'woman' or row['who'] == 'child':
            if row['pclass'] == 1 or row['pclass'] == 2:
                y = 0.9
            else:
                y = 0.7
        else:
            if row['pclass'] == 1:
                y = 0.4
            elif row['pclass'] == 2:
                y = 0.2
            else:
                y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)