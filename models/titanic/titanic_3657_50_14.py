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
        # The 'alive_yes' column is also considered assuming that if the passenger was alive, the survival probability is higher.
        # The 'alone_True' column is also considered assuming that if the passenger was alone, the survival probability might be lower.
        # The 'deck' columns are also considered assuming that passengers on certain decks might have had higher survival rates.
        # The 'embark_town' columns are also considered assuming that passengers from certain towns might have had higher survival rates.
        # The 'target' column is not considered in the prediction as it is the target variable to be predicted.

        y = 0.1 * row['sex_female'] + 0.1 * row['class_First'] + 0.1 * row['embarked_C'] - 0.1 * row['age']/80 + 0.1 * row['fare']/500 + 0.1 * row['alive_yes'] - 0.1 * row['alone_True'] + 0.1 * (row['deck_A'] + row['deck_B'] + row['deck_C'] + row['deck_D'] + row['deck_E'] + row['deck_F'] + row['deck_G'])/7 + 0.1 * (row['embark_town_Cherbourg'] + row['embark_town_Queenstown'] + row['embark_town_Southampton'])/3

        # The predicted survival probability is then normalized to be between 0 and 1.
        y = max(min(y, 1), 0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)