import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers had higher survival rates.
        # This is a simple logic and does not take into account all the features in the dataset.
        # For a more accurate prediction, a machine learning model should be trained on the dataset.

        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 0.9
        elif row['sex_female'] == 1.0 and row['class_First'] == 1.0:
            y = 0.8
        elif row['sex_female'] == 1.0:
            y = 0.7
        else:
            y = 0.2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)