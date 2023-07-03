import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We are assuming that if the passenger is female, in first class, and embarked from Cherbourg, they have a high probability of survival.
        # This is based on historical data which suggests that women and children, as well as upper-class passengers, had a higher survival rate on the Titanic.
        # We are also assuming that if the passenger is male, in third class, and embarked from Southampton, they have a low probability of survival.
        # This is based on historical data which suggests that men and lower-class passengers had a lower survival rate on the Titanic.
        # This is a very simplistic approach and would likely not perform well on unseen data.
        # A more sophisticated approach would be to use a machine learning model trained on the data.

        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 0.9
        elif row['sex_male'] == 1.0 and row['class_Third'] == 1.0 and row['embark_town_Southampton'] == 1.0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)