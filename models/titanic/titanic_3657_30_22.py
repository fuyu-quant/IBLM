import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the historical data of Titanic where female, first class passengers, and those who embarked from Cherbourg had higher survival rate
        # The age is also considered where children had higher survival rate
        # The fare is also considered where those who paid higher fare had higher survival rate
        # The values are normalized to be between 0 and 1

        y = (row['sex_female'] + row['class_First'] + row['embark_town_Cherbourg'] + (row['age'] < 18) + (row['fare'] / df['fare'].max())) / 5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)