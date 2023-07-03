import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers had higher survival rates.
        # The age of the passenger is also considered, giving higher survival probability for younger passengers.
        # The fare paid by the passenger is also considered, assuming that passengers who paid higher fares might have been given priority during the evacuation.
        # The 'sibsp' and 'parch' features are also considered, assuming that passengers with siblings/spouses or parents/children on board might have higher survival rates.
        # The 'alone' feature is also considered, assuming that passengers who were alone might have lower survival rates.
        # The 'deck' feature is also considered, assuming that passengers on certain decks might have higher survival rates.
        # The 'alive' feature is also considered, assuming that passengers who were alive might have higher survival rates.
        # The 'who' feature is also considered, assuming that passengers who were men might have lower survival rates.
        # The 'class' feature is also considered, assuming that passengers in first class might have higher survival rates.
        # The 'embark_town' feature is also considered, assuming that passengers who embarked from certain towns might have higher survival rates.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.2
        if row['pclass'] == 1.0:
            y += 0.1
        if row['embarked_C'] == 1.0:
            y += 0.1
        if row['age'] <= 18.0:
            y += 0.1
        if row['fare'] >= 50.0:
            y += 0.1
        if row['sibsp'] >= 1.0 or row['parch'] >= 1.0:
            y += 0.1
        if row['alone_False'] == 1.0:
            y += 0.1
        if row['deck_A'] == 1.0 or row['deck_B'] == 1.0 or row['deck_C'] == 1.0 or row['deck_D'] == 1.0 or row['deck_E'] == 1.0:
            y += 0.1
        if row['alive_yes'] == 1.0:
            y += 0.1
        if row['who_man'] == 0.0:
            y += 0.1
        if row['class_First'] == 1.0:
            y += 0.1
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)