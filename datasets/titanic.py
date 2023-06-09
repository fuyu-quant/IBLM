import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Please describe the process required to make the prediction below.

        # If the passenger is a female and in first or second class, predict survival
        if row['sex'] == 'female' and (row['pclass'] == 1 or row['pclass'] == 2):
            y = 1
        # If the passenger is a child (age <= 15) and in first or second class, predict survival
        elif row['age'] <= 15 and (row['pclass'] == 1 or row['pclass'] == 2):
            y = 1
        # If the passenger is a male and in third class, predict non-survival
        elif row['sex'] == 'male' and row['pclass'] == 3:
            y = 0
        # If the passenger is an adult male and traveling alone, predict non-survival
        elif row['adult_male'] and row['alone']:
            y = 0
        # For all other cases, predict non-survival
        else:
            y = 0

        y = 1 / (1 + np.exp(-y))

        output.append(y)
    return np.array(output)