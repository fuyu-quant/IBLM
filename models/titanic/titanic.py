import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        pclass = row['pclass']
        age = row['age']
        fare = row['fare']
        sex_female = row['sex_female']
        embarked_C = row['embarked_C']
        embarked_Q = row['embarked_Q']
        alive_yes = row['alive_yes']
        alone_True = row['alone_True']
        adult_male_True = row['adult_male_True']
        who_child = row['who_child']
        who_woman = row['who_woman']
        class_First = row['class_First']
        class_Second = row['class_Second']
        deck_A = row['deck_A']
        deck_B = row['deck_B']
        deck_C = row['deck_C']
        deck_D = row['deck_D']
        deck_E = row['deck_E']
        deck_F = row['deck_F']
        embark_town_Cherbourg = row['embark_town_Cherbourg']
        embark_town_Queenstown = row['embark_town_Queenstown']
        embark_town_Southampton = row['embark_town_Southampton']

        # Calculate the probability of target being 1
        probability = 0
        probability += 0.5 * sex_female
        probability += 0.3 * (class_First + class_Second)
        probability += 0.2 * (embarked_C + embarked_Q)
        probability += 0.1 * (alone_True + adult_male_True)
        probability += 0.05 * (who_child + who_woman)
        probability += 0.01 * (deck_A + deck_B + deck_C + deck_D + deck_E + deck_F)
        probability += 0.005 * (embark_town_Cherbourg + embark_town_Queenstown)

        # Normalize the probability to be between 0 and 1
        probability = min(max(probability, 0), 1)

        # Do not change the code after this point.
        output.append(probability)
    return np.array(output)