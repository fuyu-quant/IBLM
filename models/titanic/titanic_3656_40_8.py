import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # This is based on the historical data that women, children, and first-class passengers were given priority during the Titanic disaster.
        # Also, passengers who embarked from Cherbourg had a higher survival rate.
        # The age, fare, and number of siblings/spouses/parents/children are also considered in the calculation.
        # The values are normalized to be between 0 and 1.

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['embarked_C']
        y -= row['age'] / 100
        y -= row['fare'] / 500
        y -= row['sibsp'] / 10
        y -= row['parch'] / 10

        # The final probability is limited to be between 0 and 1.
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)