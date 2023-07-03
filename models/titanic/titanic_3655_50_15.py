import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority during the evacuation.
        # The embarkation point is also considered as passengers from Cherbourg had a higher survival rate.
        # The age of the passenger is also considered, giving higher survival probability for children.
        # The fare paid by the passenger is also considered, assuming that passengers who paid higher fares might have been given priority during the evacuation.
        # The number of siblings/spouses and parents/children aboard is also considered, assuming that passengers with family members might have helped each other to survive.
        # The deck of the passenger is also considered, assuming that passengers on higher decks might have had easier access to the lifeboats.
        # The 'alone' status of the passenger is also considered, assuming that passengers who were alone might have had a harder time surviving without help from family members.
        # The 'alive' status of the passenger is also considered, assuming that passengers who were alive might have had a higher survival rate.
        # The 'who' status of the passenger is also considered, assuming that men, women, and children might have had different survival rates.
        # The 'class' status of the passenger is also considered, assuming that first, second, and third class passengers might have had different survival rates.
        # The 'embark_town' status of the passenger is also considered, assuming that passengers from different embarkation towns might have had different survival rates.

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['embarked_C']
        y -= row['age'] / 80
        y += row['fare'] / 500
        y -= row['sibsp'] / 5
        y -= row['parch'] / 5
        y += row['deck_A'] / 2
        y += row['deck_B'] / 2
        y += row['deck_C'] / 2
        y += row['deck_D'] / 2
        y += row['deck_E'] / 2
        y -= row['alone_True'] / 2
        y += row['alive_yes'] / 2
        y += row['who_child'] / 2
        y += row['who_woman'] / 2
        y += row['class_Second'] / 3
        y += row['embark_town_Cherbourg'] / 3

        # Normalize the prediction to the range [0, 1]
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)