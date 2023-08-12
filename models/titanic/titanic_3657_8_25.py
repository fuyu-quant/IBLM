import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # The rules are based on the observations from the given data.
        # For example, if 'sex_female' is 1, 'alone_True' is 1, and 'class_First' is 1, 
        # then the probability of target being 1 is high.
        # Similarly, if 'sex_male' is 1, 'alone_False' is 1, and 'class_Third' is 1, 
        # then the probability of target being 1 is low.

        if row['sex_female'] == 1.0 and row['alone_True'] == 1.0 and row['class_First'] == 1.0:
            y = 0.9
        elif row['sex_male'] == 1.0 and row['alone_False'] == 1.0 and row['class_Third'] == 1.0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)