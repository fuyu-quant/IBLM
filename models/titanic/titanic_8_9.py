import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
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

        # Calculate the probability of survival based on the given features
        prob_survival = 0

        # Higher probability for female passengers
        if sex_female:
            prob_survival += 0.6

        # Higher probability for passengers in first and second class
        if pclass == 1:
            prob_survival += 0.3
        elif pclass == 2:
            prob_survival += 0.2

        # Higher probability for passengers with lower fare
        if fare <= 20:
            prob_survival += 0.2

        # Higher probability for passengers with age less than 18
        if age < 18:
            prob_survival += 0.2

        # Normalize the probability to be between 0 and 1
        prob_survival = min(1, prob_survival)

        # Do not change the code after this point.
        output.append(prob_survival)
    return np.array(output)