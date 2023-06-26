import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        pclass_weight = 0.3 if row['pclass'] == 2 else 0.1 if row['pclass'] == 3 else 0.6
        age_weight = 0.5 if row['age'] <= 27 else 0.3 if row['age'] <= 29 else 0.2
        fare_weight = 0.4 if row['fare'] <= 10.5 else 0.3 if row['fare'] <= 26.0 else 0.2 if row['fare'] <= 26.25 else 0.1
        sex_weight = 0.6 if row['sex_female'] else 0.4
        embarked_weight = 0.5 if row['embarked_S'] else 0.3 if row['embarked_C'] else 0.2
        alive_weight = 0.6 if row['alive_yes'] else 0.4
        alone_weight = 0.5 if row['alone_True'] else 0.5
        adult_male_weight = 0.6 if row['adult_male_True'] else 0.4
        who_weight = 0.4 if row['who_child'] else 0.3 if row['who_man'] else 0.3
        class_weight = 0.4 if row['class_Second'] else 0.3 if row['class_Third'] else 0.3
        embark_town_weight = 0.5 if row['embark_town_Southampton'] else 0.3 if row['embark_town_Cherbourg'] else 0.2
        
        # Calculate the final probability
        y = pclass_weight * age_weight * fare_weight * sex_weight * embarked_weight * alive_weight * alone_weight * adult_male_weight * who_weight * class_weight * embark_town_weight

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)