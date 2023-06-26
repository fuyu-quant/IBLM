import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        pclass_weight = 0.3 if row['pclass'] == 2 else 0.1
        age_weight = 0.5 if row['age'] <= 27 else 0.2
        sibsp_weight = 0.4 if row['sibsp'] == 0 else 0.1
        parch_weight = 0.4 if row['parch'] == 0 else 0.1
        fare_weight = 0.5 if row['fare'] <= 10.5 else 0.2
        sex_female_weight = 0.6 if row['sex_female'] else 0.1
        embarked_S_weight = 0.5 if row['embarked_S'] else 0.1
        alive_yes_weight = 0.6 if row['alive_yes'] else 0.1
        alone_True_weight = 0.5 if row['alone_True'] else 0.1
        adult_male_True_weight = 0.5 if row['adult_male_True'] else 0.1
        class_Third_weight = 0.4 if row['class_Third'] else 0.1
        embark_town_Southampton_weight = 0.5 if row['embark_town_Southampton'] else 0.1

        # Calculate the final probability
        y = (pclass_weight + age_weight + sibsp_weight + parch_weight + fare_weight + sex_female_weight + embarked_S_weight + alive_yes_weight + alone_True_weight + adult_male_True_weight + class_Third_weight + embark_town_Southampton_weight) / 12

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)