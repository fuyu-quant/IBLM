import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, passengers in first class (pclass=1) or female passengers (sex_female=1) had higher survival rates.
        # Similarly, passengers who were alone (alone_True=1) or who embarked at Cherbourg (embark_town_Cherbourg=1) also had higher survival rates.
        # The weights for these features are chosen based on their relative importance in determining survival.
        y = 0.3*row['pclass'] + 0.5*row['sex_female'] + 0.1*row['alone_True'] + 0.1*row['embark_town_Cherbourg']

        # Normalize the prediction to be between 0 and 1
        y = (y - df[['pclass', 'sex_female', 'alone_True', 'embark_town_Cherbourg']].min().sum()) / (df[['pclass', 'sex_female', 'alone_True', 'embark_town_Cherbourg']].max().sum() - df[['pclass', 'sex_female', 'alone_True', 'embark_town_Cherbourg']].min().sum())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)