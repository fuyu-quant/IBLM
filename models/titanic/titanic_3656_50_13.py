import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, 'sex_female', 'fare', 'class_First', 'who_child' are given more weightage as they are more likely to result in survival.
        # Similarly, 'sex_male', 'pclass', 'alone_True', 'class_Third' are given less weightage as they are less likely to result in survival.
        y = 0.3*row['sex_female'] + 0.2*row['fare'] + 0.15*row['class_First'] + 0.15*row['who_child'] - 0.1*row['sex_male'] - 0.1*row['pclass'] - 0.05*row['alone_True'] - 0.05*row['class_Third']

        # Normalizing the output to be between 0 and 1
        y = (y - df.min()) / (df.max() - df.min())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)