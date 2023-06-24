import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given features
        pclass_factor = 0.9 if row['pclass'] == 1 else (0.7 if row['pclass'] == 2 else 0.5)
        age_factor = 1.0 if row['age'] <= 15 else (0.8 if row['age'] <= 30 else (0.6 if row['age'] <= 45 else 0.4))
        fare_factor = 0.9 if row['fare'] >= 50 else (0.7 if row['fare'] >= 20 else 0.5)
        sex_factor = 0.8 if row['sex_female'] else 0.6
        embarked_factor = 0.8 if row['embarked_C'] else (0.7 if row['embarked_Q'] else 0.6)
        alone_factor = 0.7 if row['alone_True'] else 0.6
        class_factor = 0.9 if row['class_First'] else (0.7 if row['class_Second'] else 0.5)
        deck_factor = 0.9 if row['deck_A'] else (0.8 if row['deck_B'] else (0.7 if row['deck_C'] else (0.6 if row['deck_D'] else (0.5 if row['deck_E'] else (0.4 if row['deck_F'] else 0.3)))))
        embark_town_factor = 0.8 if row['embark_town_Cherbourg'] else (0.7 if row['embark_town_Queenstown'] else 0.6)

        # Calculate the final probability
        y = pclass_factor * age_factor * fare_factor * sex_factor * embarked_factor * alone_factor * class_factor * deck_factor * embark_town_factor

        # Normalize the probability to be between 0 and 1
        y = (y - 0.5) / 2.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)