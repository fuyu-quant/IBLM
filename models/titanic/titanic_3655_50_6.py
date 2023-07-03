import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, 'pclass' is inversely proportional to survival rate, 'sex_female' has higher survival rate, 'fare' is directly proportional to survival rate, etc.
        # The weights for these features are determined based on their importance in survival.
        y = 0.1*row['pclass'] + 0.3*row['sex_female'] + 0.2*row['fare'] + 0.1*row['embarked_C'] + 0.1*row['alone_False'] + 0.1*row['adult_male_False'] + 0.1*row['class_First']

        # Normalize the output to be between 0 and 1
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)