import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, we can see that the survival rate is higher for females, children, first class passengers, and those who embarked from Cherbourg.
        # We can also see that the survival rate is lower for males, adults, third class passengers, and those who embarked from Southampton.
        # Therefore, we can create a simple rule-based model to predict the survival probability based on these observations.

        # Initialize the probability to 0.5 (neutral)
        prob = 0.5

        # Increase the probability if the passenger is female, a child, or a first class passenger
        if row['sex_female'] == 1.0 or row['who_child'] == 1.0 or row['class_First'] == 1.0:
            prob += 0.1

        # Increase the probability if the passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            prob += 0.05

        # Decrease the probability if the passenger is male, an adult, or a third class passenger
        if row['sex_male'] == 1.0 or row['who_man'] == 1.0 or row['class_Third'] == 1.0:
            prob -= 0.1

        # Decrease the probability if the passenger embarked from Southampton
        if row['embark_town_Southampton'] == 1.0:
            prob -= 0.05

        # Ensure the probability is within the range [0, 1]
        prob = max(0, min(1, prob))

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)