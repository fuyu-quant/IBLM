import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if 'pclass' is 1.0 and 'sex_female' is 1.0, the target is likely to be 1.
        # Similarly, if 'pclass' is 3.0 and 'sex_male' is 1.0, the target is likely to be 0.
        # These rules are not perfect and may not work well on unseen data.
        # For a more accurate prediction, a machine learning model should be trained on the data.

        if row['pclass'] == 1.0 and row['sex_female'] == 1.0:
            y = 1.0
        elif row['pclass'] == 3.0 and row['sex_male'] == 1.0:
            y = 0.0
        else:
            y = 0.5  # If none of the rules apply, we predict a probability of 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)