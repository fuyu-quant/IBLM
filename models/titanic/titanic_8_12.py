import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        pclass_factor = 0.5 if row['pclass'] == 2 else 0.3 if row['pclass'] == 3 else 0.2
        age_factor = 0.5 if row['age'] <= 27 else 0.3 if row['age'] <= 29 else 0.2
        sibsp_factor = 0.5 if row['sibsp'] == 0 else 0.3 if row['sibsp'] == 1 else 0.2
        parch_factor = 0.5 if row['parch'] == 0 else 0.3 if row['parch'] == 2 else 0.2
        fare_factor = 0.5 if row['fare'] <= 10.5 else 0.3 if row['fare'] <= 26.25 else 0.2
        sex_female_factor = 0.5 if row['sex_female'] else 0.3
        sex_male_factor = 0.5 if row['sex_male'] else 0.3
        embarked_S_factor = 0.5 if row['embarked_S'] else 0.3
        alive_yes_factor = 0.5 if row['alive_yes'] else 0.3
        alone_True_factor = 0.5 if row['alone_True'] else 0.3
        adult_male_True_factor = 0.5 if row['adult_male_True'] else 0.3
        who_man_factor = 0.5 if row['who_man'] else 0.3
        class_Third_factor = 0.5 if row['class_Third'] else 0.3
        embark_town_Southampton_factor = 0.5 if row['embark_town_Southampton'] else 0.3

        # Calculate the final probability
        y = pclass_factor * age_factor * sibsp_factor * parch_factor * fare_factor * sex_female_factor * sex_male_factor * embarked_S_factor * alive_yes_factor * alone_True_factor * adult_male_True_factor * who_man_factor * class_Third_factor * embark_town_Southampton_factor

        # Normalize the probability to be between 0 and 1
        y = y / (y + (1 - y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)