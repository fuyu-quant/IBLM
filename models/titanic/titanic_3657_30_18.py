import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority for lifeboats.
        # The age of the passenger is also considered, giving higher survival probability for younger passengers.
        # The fare paid by the passenger is also considered, assuming that passengers who paid higher fares might have been given priority for lifeboats.
        # The 'sibsp' and 'parch' features are also considered, assuming that passengers with siblings/spouses or parents/children on board might have higher survival probability.
        # The 'alone' feature is also considered, assuming that passengers who were alone might have lower survival probability.
        # The 'deck' feature is also considered, assuming that passengers on higher decks (closer to the lifeboats) might have higher survival probability.

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['embarked_C']
        y += row['age'] * -0.01
        y += row['fare'] * 0.001
        y += row['sibsp'] * 0.1
        y += row['parch'] * 0.1
        y += row['alone_False'] * 0.1
        y += (row['deck_A'] + row['deck_B'] + row['deck_C'] + row['deck_D'] + row['deck_E']) * 0.1

        # Normalize the prediction to be between 0 and 1
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)