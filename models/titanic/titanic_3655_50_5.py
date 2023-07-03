import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, 'pclass' is negatively correlated with survival, so we subtract it.
        # On the other hand, 'fare' and 'sex_female' are positively correlated with survival, so we add them.
        # We also consider 'age', 'sibsp', 'parch' and 'embarked_C' as they might have some influence on the survival.
        # This is a very basic model and might not give very accurate results.
        y = row['fare'] + row['sex_female'] - row['pclass'] + row['age']*0.1 - row['sibsp']*0.1 - row['parch']*0.1 + row['embarked_C']*0.1

        # Normalize the output to be between 0 and 1
        y = (y - df.min().min()) / (df.max().max() - df.min().min())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)