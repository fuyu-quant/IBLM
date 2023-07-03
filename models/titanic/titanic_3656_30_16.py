import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, belong to first class, embarked from Cherbourg, and are alone.
        # These conditions are chosen based on the general survival statistics of the Titanic disaster.
        # This is a very basic logic and can be improved with more complex conditions and machine learning models.
        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.3
        if row['embarked_C'] == 1.0:
            y += 0.2
        if row['alone_True'] == 1.0:
            y += 0.2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)