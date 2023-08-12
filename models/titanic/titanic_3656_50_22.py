import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that we are giving more weightage to the features which are more likely to result in survival.
        # For example, if the passenger is a female (sex_female=1), embarked from Cherbourg (embarked_C=1), travelling alone (alone_True=1), 
        # and in first class (class_First=1), the chances of survival are high.
        # Similarly, if the passenger is a male (sex_male=1), embarked from Southampton (embark_town_Southampton=1), not alone (alone_False=1), 
        # and in third class (class_Third=1), the chances of survival are low.
        # The age and fare are also considered, younger and higher fare passengers are considered to have higher chances of survival.
        # The weights for these features are determined based on their perceived importance.

        y = 0.3*row['sex_female'] + 0.1*row['embarked_C'] + 0.1*row['alone_True'] + 0.2*row['class_First'] - 0.3*row['sex_male'] - 0.1*row['embark_town_Southampton'] - 0.1*row['alone_False'] - 0.2*row['class_Third'] + 0.05*row['age'] + 0.05*row['fare']

        # The output is then normalized to be between 0 and 1 using the sigmoid function.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)