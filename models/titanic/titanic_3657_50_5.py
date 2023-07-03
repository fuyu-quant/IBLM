import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children and first class passengers had higher survival rates.
        # The age factor is also considered, giving higher survival probability for children.
        # The fare is also considered, assuming that passengers who paid higher fares might have been given priority during the rescue.
        # The 'alive_yes' column is also considered assuming that if a passenger was marked as alive, they have higher chances of survival.
        # The 'deck' columns are not considered in this logic as the deck information might not be available for all passengers.
        # The 'sibsp' and 'parch' columns are also not considered in this logic as they might not directly influence the survival probability.
        # The 'alone' columns are also not considered in this logic as being alone or not might not directly influence the survival probability.
        # The 'who' columns are also not considered in this logic as they are similar to the 'sex' and 'age' columns.
        # The 'class' columns are also not considered in this logic as they are similar to the 'pclass' column.
        # The 'embark_town' columns are also not considered in this logic as they are similar to the 'embarked' columns.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.2
        if row['embarked_C'] == 1.0:
            y += 0.1
        if row['age'] <= 18.0:
            y += 0.1
        if row['fare'] > 50.0:
            y += 0.1
        if row['alive_yes'] == 1.0:
            y += 0.2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)