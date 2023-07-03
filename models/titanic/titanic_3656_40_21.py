import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority for lifeboats.
        # The conditions can be adjusted based on the specific dataset and the correlations between the features and the target variable.

        prob = 0.0
        if row['sex_female'] == 1.0:
            prob += 0.3
        if row['class_First'] == 1.0:
            prob += 0.3
        if row['embark_town_Cherbourg'] == 1.0:
            prob += 0.2

        # Age is also a significant factor. Younger passengers have higher survival rate.
        if row['age'] <= 10.0:
            prob += 0.2
        elif row['age'] <= 30.0:
            prob += 0.1

        # Normalize the probability to make sure it's between 0 and 1.
        prob = min(max(prob, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)