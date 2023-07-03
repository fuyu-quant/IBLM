import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority during the evacuation.
        # The 'embarked_C' condition is based on the observation that passengers who embarked from Cherbourg had a higher survival rate.
        # The 'fare' condition is based on the assumption that passengers who paid a higher fare had a higher survival rate.
        # The 'age' condition is based on the assumption that younger passengers had a higher survival rate.
        # The 'sibsp' and 'parch' conditions are based on the assumption that passengers with siblings/spouses or parents/children aboard had a higher survival rate.
        # The 'alone_True' condition is based on the assumption that passengers who were alone had a lower survival rate.

        y = 0.5  # base probability

        if row['sex_female'] == 1.0:
            y += 0.3
        if row['class_First'] == 1.0:
            y += 0.1
        if row['embarked_C'] == 1.0:
            y += 0.05
        if row['fare'] > 30.0:
            y += 0.05
        if row['age'] < 18.0:
            y += 0.05
        if row['sibsp'] > 0.0 or row['parch'] > 0.0:
            y += 0.05
        if row['alone_True'] == 1.0:
            y -= 0.05

        # Ensure the probability is within the range [0, 1]
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)