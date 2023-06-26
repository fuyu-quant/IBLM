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

        # Calculate probability of target being 1
        prob = 0
        if sex_female:
            prob += 0.3
        if pclass == 1:
            prob += 0.2
        if fare > 50:
            prob += 0.1
        if age < 18:
            prob += 0.1
        if embarked_C:
            prob += 0.05
        if alive_yes:
            prob += 0.1
        if alone_False:
            prob += 0.05
        if adult_male_False:
            prob += 0.05
        if who_child:
            prob += 0.05
        if class_First:
            prob += 0.05
        if deck_B or deck_C or deck_D or deck_E:
            prob += 0.05
        if embark_town_Cherbourg:
            prob += 0.05

        # Limit probability to range [0, 1]
        prob = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)