import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for passengers who are female, in first class, and embarked from Cherbourg
        # These factors are chosen based on the historical data of the Titanic disaster where females, first class passengers, and passengers from Cherbourg had higher survival rates
        y = 0.3*row['sex_female'] + 0.3*row['class_First'] + 0.3*row['embark_town_Cherbourg'] + 0.1*row['fare']

        # Normalizing the output to be between 0 and 1
        y = (y - df.min()) / (df.max() - df.min())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)