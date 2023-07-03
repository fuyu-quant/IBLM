import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are usually the people who have higher survival rate based on historical data
        # The age, fare, and number of siblings/spouses/parents/children are also considered
        # The younger, the higher fare, and the less number of siblings/spouses/parents/children, the higher the survival rate
        y = 0.5
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['class_First'] == 1.0:
            y += 0.1
        if row['embarked_C'] == 1.0:
            y += 0.1
        if row['age'] <= 30.0:
            y += 0.05
        if row['fare'] >= 30.0:
            y += 0.05
        if row['sibsp'] == 0.0 and row['parch'] == 0.0:
            y += 0.05

        # Limit the probability between 0 and 1
        y = min(max(y, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)