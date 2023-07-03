import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We are assuming that if the passenger is female, has a first class ticket, and embarked from Cherbourg, they have a high probability of survival.
        # This is based on historical data which suggests that women, children, and upper-class passengers were more likely to survive the Titanic disaster.
        # This is a very simplistic approach and a real-world solution would likely use a machine learning model to make more accurate predictions.

        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 0.9
        else:
            y = 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)