import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, 'sex_female', 'fare', 'class_First', 'who_woman' are given more weightage as they are more likely to result in survival.
        # Similarly, 'sex_male', 'age', 'class_Third', 'who_man' are given less weightage as they are less likely to result in survival.
        # The weights are arbitrary and can be adjusted for better accuracy.
        y = 0.3*row['sex_female'] + 0.2*row['fare'] + 0.2*row['class_First'] + 0.2*row['who_woman'] - 0.1*row['sex_male'] - 0.1*row['age'] - 0.1*row['class_Third'] - 0.1*row['who_man']

        # Normalize the output to be between 0 and 1
        y = (y - df.min()) / (df.max() - df.min())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)