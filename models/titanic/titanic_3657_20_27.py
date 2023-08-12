import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are usually the people who have higher survival rate based on the Titanic dataset
        # The values are normalized to be between 0 and 1
        y = (row['sex_female'] + row['class_First'] + row['embark_town_Cherbourg']) / 3

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)