import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if 'sex_female' is 1, 'alive_yes' is 1 and 'class_First' is 1, 
        # then the probability of target being 1 is high.
        # Similarly, if 'sex_male' is 1, 'alive_no' is 1 and 'class_Third' is 1, 
        # then the probability of target being 0 is high.
        # These rules are not perfect and may not work well on unseen data.

        if row['sex_female'] == 1.0 and row['alive_yes'] == 1.0 and row['class_First'] == 1.0:
            y = 1.0
        elif row['sex_male'] == 1.0 and row['alive_no'] == 1.0 and row['class_Third'] == 1.0:
            y = 0.0
        else:
            y = 0.5  # If none of the above conditions are met, we assign a neutral probability.

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)