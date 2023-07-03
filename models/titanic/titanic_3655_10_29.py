import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based system to predict the target.
        # The rules are based on the data provided and may not be accurate for all cases.
        # This is a placeholder for a more sophisticated model.

        # Rule 1: If the passenger is female, there is a high probability of survival.
        if row['sex_female'] == 1:
            y = 0.8
        # Rule 2: If the passenger is male and in first class, there is a moderate probability of survival.
        elif row['sex_male'] == 1 and row['class_First'] == 1:
            y = 0.6
        # Rule 3: If the passenger is male and in second or third class, there is a low probability of survival.
        elif row['sex_male'] == 1 and (row['class_Second'] == 1 or row['class_Third'] == 1):
            y = 0.3
        # Rule 4: If none of the above conditions are met, assign a default probability.
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)