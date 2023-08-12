import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers had higher survival rates.
        # The conditions are arbitrarily weighted for the purpose of this task.
        y = 0.3*row['sex_female'] + 0.3*row['class_First'] + 0.2*row['embark_town_Cherbourg'] + 0.1*row['fare'] + 0.1*row['age']

        # Normalize the output to be between 0 and 1
        y = (y - df.min()) / (df.max() - df.min())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)