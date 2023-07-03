import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # This is based on the historical data that women, children, and the upper-class passengers were given priority during the evacuation.
        # The age, fare, and number of siblings/spouses/parents/children are also considered.
        # The logic can be refined further based on more detailed analysis of the data.

        p = 0.0
        if row['sex_female'] == 1.0:
            p += 0.3
        if row['pclass'] == 1.0:
            p += 0.2
        if row['embarked_C'] == 1.0:
            p += 0.1
        if row['age'] <= 10.0:
            p += 0.1
        if row['fare'] >= 50.0:
            p += 0.1
        if row['sibsp'] <= 2.0:
            p += 0.1
        if row['parch'] <= 2.0:
            p += 0.1

        # Normalize the probability to the range [0, 1]
        p = min(max(p, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(p)
    return np.array(output)