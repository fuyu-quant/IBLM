import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority for lifeboats.
        # The port of embarkation is also considered as passengers from Cherbourg had a higher survival rate.
        # The age of the passenger is also considered, giving higher probability for survival for children.
        # The fare paid by the passenger is also considered, assuming that passengers who paid higher fares had higher survival rates.
        # This is a simple heuristic and does not take into account interactions between variables.

        y = 0.0
        if row['sex_female'] == 1:
            y += 0.3
        if row['pclass'] == 1:
            y += 0.3
        if row['embarked_C'] == 1:
            y += 0.1
        if row['age'] < 18:
            y += 0.2
        if row['fare'] > df['fare'].median():
            y += 0.1

        # Normalize the probability to be between 0 and 1
        y = min(max(y, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)