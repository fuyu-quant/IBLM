import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is that we are giving more weightage to the features which are more likely to result in survival.
        # For example, if the passenger is female, if the passenger is in first class, if the passenger is a child, 
        # if the passenger embarked from Cherbourg, if the passenger is alone, and if the passenger's deck is B, C, D, or E.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first class passengers 
        # had higher survival rates. Also, passengers from Cherbourg had a higher survival rate and passengers on certain decks had 
        # easier access to lifeboats.
        # The weights for these conditions are assumed and can be adjusted for better accuracy.
        
        y = 0.0
        if row['sex_female'] == 1:
            y += 0.2
        if row['pclass'] == 1:
            y += 0.2
        if row['who_child'] == 1:
            y += 0.2
        if row['embarked_C'] == 1:
            y += 0.1
        if row['alone_True'] == 1:
            y += 0.1
        if row['deck_B'] == 1 or row['deck_C'] == 1 or row['deck_D'] == 1 or row['deck_E'] == 1:
            y += 0.2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)