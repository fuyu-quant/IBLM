import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given features
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

        # Define the weights for each feature
        weights = {
            'pclass': -0.15,
            'age': -0.005,
            'sibsp': -0.05,
            'parch': -0.03,
            'fare': 0.002,
            'sex_female': 0.3,
            'sex_male': -0.3,
            'embarked_C': 0.1,
            'embarked_Q': 0.05,
            'embarked_S': -0.1,
            'alive_no': -0.5,
            'alive_yes': 0.5,
            'alone_False': -0.1,
            'alone_True': 0.1,
            'adult_male_False': 0.2,
            'adult_male_True': -0.2,
            'who_child': 0.2,
            'who_man': -0.2,
            'who_woman': 0.2,
            'class_First': 0.2,
            'class_Second': 0.1,
            'class_Third': -0.2,
            'deck_A': 0.05,
            'deck_B': 0.1,
            'deck_C': 0.05,
            'deck_D': 0.1,
            'deck_E': 0.1,
            'deck_F': 0.05,
            'deck_G': 0.05,
            'embark_town_Cherbourg': 0.1,
            'embark_town_Queenstown': 0.05,
            'embark_town_Southampton': -0.1
        }

        # Calculate the probability of target being 1
        y = 1 / (1 + np.exp(-(
            pclass * weights['pclass'] +
            age * weights['age'] +
            sibsp * weights['sibsp'] +
            parch * weights['parch'] +
            fare * weights['fare'] +
            sex_female * weights['sex_female'] +
            sex_male * weights['sex_male'] +
            embarked_C * weights['embarked_C'] +
            embarked_Q * weights['embarked_Q'] +
            embarked_S * weights['embarked_S'] +
            alive_no * weights['alive_no'] +
            alive_yes * weights['alive_yes'] +
            alone_False * weights['alone_False'] +
            alone_True * weights['alone_True'] +
            adult_male_False * weights['adult_male_False'] +
            adult_male_True * weights['adult_male_True'] +
            who_child * weights['who_child'] +
            who_man * weights['who_man'] +
            who_woman * weights['who_woman'] +
            class_First * weights['class_First'] +
            class_Second * weights['class_Second'] +
            class_Third * weights['class_Third'] +
            deck_A * weights['deck_A'] +
            deck_B * weights['deck_B'] +
            deck_C * weights['deck_C'] +
            deck_D * weights['deck_D'] +
            deck_E * weights['deck_E'] +
            deck_F * weights['deck_F'] +
            deck_G * weights['deck_G'] +
            embark_town_Cherbourg * weights['embark_town_Cherbourg'] +
            embark_town_Queenstown * weights['embark_town_Queenstown'] +
            embark_town_Southampton * weights['embark_town_Southampton']
        )))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)