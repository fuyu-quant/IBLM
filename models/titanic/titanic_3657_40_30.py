import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers had higher survival rates.
        # The age of the passenger is also considered, giving higher survival probability for children.
        # The fare paid by the passenger is also considered, assuming that passengers who paid higher fares might have been given priority during the evacuation.
        # The 'alive_yes' column is also considered assuming that if the passenger was marked as alive, they survived the disaster.
        # The 'alone_True' column is also considered assuming that if the passenger was alone, they had less chances of survival.
        # The 'deck' columns are not considered in this logic as the deck information might not be available for all passengers.
        # The 'embark_town' columns are not considered in this logic as the town of embarkation might not have a significant impact on the survival of the passengers.
        # The 'sibsp' and 'parch' columns are not considered in this logic as the number of siblings/spouses and parents/children might not have a significant impact on the survival of the passengers.
        # The 'class' columns are not considered in this logic as the passenger class information is already considered through the 'pclass' column.
        # The 'who' columns are not considered in this logic as the gender and age of the passengers are already considered through the 'sex' and 'age' columns.
        # The 'adult_male' columns are not considered in this logic as the gender and age of the passengers are already considered through the 'sex' and 'age' columns.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.2
        if row['embarked_C'] == 1.0:
            y += 0.1
        if row['age'] <= 18.0:
            y += 0.1
        if row['fare'] >= 50.0:
            y += 0.1
        if row['alive_yes'] == 1.0:
            y += 0.1
        if row['alone_True'] == 0.0:
            y += 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)