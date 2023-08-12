import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are usually the people who have higher survival rate based on the data
        # The age, fare, and number of siblings/spouses/parents/children are also considered
        # The younger, the higher fare, and the less number of siblings/spouses/parents/children, the higher the survival rate
        y = 0.3 * row['sex_female'] + 0.2 * row['class_First'] + 0.1 * row['embark_town_Cherbourg'] - 0.1 * row['age']/80 - 0.1 * row['fare']/500 - 0.1 * (row['sibsp'] + row['parch'])/10

        # The probability is then normalized to be between 0 and 1
        y = (y + 1) / 2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)