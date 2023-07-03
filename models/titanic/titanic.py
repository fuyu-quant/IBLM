import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # as these factors are generally associated with higher survival rates in the Titanic disaster.
        # The age, fare, and number of siblings/spouses/parents/children aboard are also considered.
        # The values are normalized to be between 0 and 1.

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['embark_town_Cherbourg']
        y -= row['age'] / 100.0
        y -= row['fare'] / 500.0
        y -= row['sibsp'] / 10.0
        y -= row['parch'] / 10.0

        # Ensure the probability is between 0 and 1
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)