import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are usually the people who have higher survival rate based on historical data
        # The age, fare, and number of siblings/spouses/parents/children aboard are also considered
        # The younger, the higher fare, and the less number of siblings/spouses/parents/children usually have higher survival rate
        # The values are normalized to be between 0 and 1 by dividing by the maximum value of each column
        y = (row['sex_female'] + row['class_First'] + row['embark_town_Cherbourg'] + 
             (1 - row['age']/df['age'].max()) + row['fare']/df['fare'].max() + 
             (1 - row['sibsp']/df['sibsp'].max()) + (1 - row['parch']/df['parch'].max())) / 7

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)