import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are just assumptions based on general knowledge about the Titanic disaster, where women, children and first class passengers had higher survival rate
        # The age is also considered, giving higher probability for children
        # The fare is also considered, assuming that higher fare might indicate higher class or more resources, thus higher survival rate
        # The values are normalized to be between 0 and 1 by dividing by the maximum value of each factor
        y = (row['sex_female'] + row['class_First'] + row['embark_town_Cherbourg'] + row['age']/df['age'].max() + row['fare']/df['fare'].max()) / 5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)