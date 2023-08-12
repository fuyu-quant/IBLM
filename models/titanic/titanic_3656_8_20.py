import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We assume that if the passenger is female, in first class, and embarked from Cherbourg, they have a high probability of survival.
        # This is based on historical data from the Titanic disaster, where women, children, and first-class passengers were given priority for lifeboats.
        # We also consider the age of the passenger, with younger passengers being more likely to survive.
        # This is a very simplistic model and would likely not perform well on real-world data.

        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0 and row['age'] < 30:
            y = 0.9
        elif row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 0.8
        elif row['sex_female'] == 1.0 and row['class_First'] == 1.0:
            y = 0.7
        elif row['sex_female'] == 1.0:
            y = 0.6
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)