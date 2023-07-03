import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for target=1 if the passenger is female, in first class, and embarked from Cherbourg.
        # This is based on the observation that females, first class passengers, and those who embarked from Cherbourg had higher survival rates.
        # The age, fare, and number of siblings/spouses/parents/children are also considered, with younger, higher-paying, and less accompanied passengers given higher probabilities.
        # The probabilities are normalized to be between 0 and 1.
        y = (row['sex_female'] + row['class_First'] + row['embark_town_Cherbourg'] + max(0, (30 - row['age'])/30) + max(0, (100 - row['fare'])/100) + max(0, (3 - row['sibsp'])/3) + max(0, (3 - row['parch'])/3))/7

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)