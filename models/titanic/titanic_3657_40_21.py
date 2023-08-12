import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, if the passenger is a female (sex_female=1), embarked from Cherbourg (embarked_C=1), travelling in first class (class_First=1), and travelling alone (alone_True=1), 
        # then the chances of survival are high. Similarly, if the passenger is a male (sex_male=1), embarked from Southampton (embark_town_Southampton=1), travelling in third class (class_Third=1), 
        # and not travelling alone (alone_False=1), then the chances of survival are low.
        # The weights for these features are determined based on their importance in determining the survival of the passenger.

        y = 0.3*row['sex_female'] + 0.2*row['embarked_C'] + 0.2*row['class_First'] + 0.1*row['alone_True'] - 0.3*row['sex_male'] - 0.2*row['embark_town_Southampton'] - 0.2*row['class_Third'] - 0.1*row['alone_False']

        # The output is then normalized to be between 0 and 1 using the sigmoid function.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)