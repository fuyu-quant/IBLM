import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Prediction logic
        pclass = row['pclass']
        age = row['age']
        sibsp = row['sibsp']
        parch = row['parch']
        fare = row['fare']
        sex_female = row['sex_female']
        sex_male = row['sex_male']
        embarked_C = row['embarked_C']
        embarked_Q = row['embarked_Q']
        embarked_S = row['embarked_S']
        alive_no = row['alive_no']
        alive_yes = row['alive_yes']
        alone_False = row['alone_False']
        alone_True = row['alone_True']
        adult_male_False = row['adult_male_False']
        adult_male_True = row['adult_male_True']
        who_child = row['who_child']
        who_man = row['who_man']
        who_woman = row['who_woman']
        class_First = row['class_First']
        class_Second = row['class_Second']
        class_Third = row['class_Third']
        deck_A = row['deck_A']
        deck_B = row['deck_B']
        deck_C = row['deck_C']
        deck_D = row['deck_D']
        deck_E = row['deck_E']
        deck_F = row['deck_F']
        deck_G = row['deck_G']
        embark_town_Cherbourg = row['embark_town_Cherbourg']
        embark_town_Queenstown = row['embark_town_Queenstown']
        embark_town_Southampton = row['embark_town_Southampton']

        # Calculate probability based on features
        prob = 0
        prob += 0.5 * sex_female
        prob += 0.3 * (class_First + class_Second)
        prob -= 0.2 * class_Third
        prob += 0.1 * (embarked_C + embarked_Q)
        prob -= 0.1 * embarked_S
        prob += 0.2 * (age < 18)
        prob -= 0.1 * (age > 60)
        prob += 0.1 * (fare > np.mean(fare))
        prob -= 0.1 * (fare < np.mean(fare))
        prob += 0.2 * (sibsp + parch > 0)
        prob -= 0.1 * (sibsp + parch == 0)

        # Clip probability between 0 and 1
        y = np.clip(prob, 0, 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)