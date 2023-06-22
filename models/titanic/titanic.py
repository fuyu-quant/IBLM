import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Prediction logic
        pclass = row['pclass']
        age = row['age']
        fare = row['fare']
        sex_female = row['sex_female']
        embarked_C = row['embarked_C']
        embarked_Q = row['embarked_Q']
        embarked_S = row['embarked_S']
        alive_yes = row['alive_yes']
        alone_True = row['alone_True']
        adult_male_True = row['adult_male_True']
        who_child = row['who_child']
        who_woman = row['who_woman']
        class_First = row['class_First']
        class_Second = row['class_Second']
        class_Third = row['class_Third']
        embark_town_Cherbourg = row['embark_town_Cherbourg']
        embark_town_Queenstown = row['embark_town_Queenstown']
        embark_town_Southampton = row['embark_town_Southampton']

        # Calculate probability based on the given data
        prob = 0
        if sex_female:
            prob += 0.6
        if pclass == 1:
            prob += 0.3
        elif pclass == 2:
            prob += 0.1
        if age <= 16:
            prob += 0.2
        if fare > 50:
            prob += 0.1
        if embarked_C:
            prob += 0.1
        if alive_yes:
            prob += 0.3
        if alone_True:
            prob -= 0.1
        if adult_male_True:
            prob -= 0.2
        if who_child:
            prob += 0.2
        if who_woman:
            prob += 0.1
        if class_First:
            prob += 0.2
        if class_Second:
            prob += 0.1
        if embark_town_Cherbourg:
            prob += 0.1
        if embark_town_Queenstown:
            prob += 0.05
        if embark_town_Southampton:
            prob += 0.05

        # Normalize probability to be between 0 and 1
        y = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)