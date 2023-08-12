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
        # The number of siblings/spouses and parents/children aboard is also considered, assuming that passengers with family members might have helped each other to survive.
        # The deck of the passenger is also considered, assuming that passengers on higher decks had a higher chance of survival as they were closer to the lifeboats.

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['embark_town_Cherbourg']
        y -= row['age'] / 80
        y += row['fare'] / 500
        y += row['sibsp'] / 8
        y += row['parch'] / 6
        y += row['deck_A'] / 2
        y += row['deck_B'] / 2
        y += row['deck_C'] / 2
        y += row['deck_D'] / 2
        y += row['deck_E'] / 2
        y += row['deck_F'] / 2
        y += row['deck_G'] / 2

        # Normalize the prediction to the range [0, 1]
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)