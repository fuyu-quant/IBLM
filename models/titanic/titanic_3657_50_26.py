import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, 'sex_female', 'fare', 'class_First', 'who_woman' are given more weightage as they are more likely to result in survival.
        # Similarly, 'sex_male', 'pclass', 'class_Third', 'who_man' are given less weightage as they are less likely to result in survival.
        # The weights are arbitrary and can be tuned for better results.
        y = 0.3*row['sex_female'] + 0.2*row['fare'] + 0.2*row['class_First'] + 0.2*row['who_woman'] - 0.2*row['sex_male'] - 0.2*row['pclass'] - 0.2*row['class_Third'] - 0.2*row['who_man']

        # Converting the result to a probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)