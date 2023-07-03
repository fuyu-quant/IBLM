import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, 
        # belong to first class, embarked from Cherbourg and are adults. 
        # These conditions are based on the historical data of the Titanic disaster where 
        # women, first class passengers and those who embarked from Cherbourg had higher survival rates.
        # The age factor is also considered where younger passengers had slightly higher survival rates.
        # The 'fare' is also considered as passengers who paid higher fares had higher survival rates.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['class_First'] == 1.0:
            y += 0.2
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.1
        if row['who_adult'] == 1.0:
            y += 0.1
        if row['age'] <= 30.0:
            y += 0.1
        if row['fare'] >= 30.0:
            y += 0.2

        # Normalize the probability to be between 0 and 1
        y = min(max(y, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)