import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, in first class, and embarked from Cherbourg
        # and lower probability for the passengers who are male, in third class, and embarked from Southampton.
        # The age, number of siblings/spouses, number of parents/children, and fare are also considered in the prediction.
        # The younger the passenger and the higher the fare, the higher the probability.
        # The more siblings/spouses and parents/children the passenger has, the lower the probability.
        y = (row['sex_female'] + row['class_First'] + row['embark_town_Cherbourg'] + row['age'] / 100 + row['fare'] / 100
             - row['sex_male'] - row['class_Third'] - row['embark_town_Southampton'] - row['sibsp'] / 10 - row['parch'] / 10)

        # Normalize the prediction to the range [0, 1]
        y = (y + 3) / 6

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)