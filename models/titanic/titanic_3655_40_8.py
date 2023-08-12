import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is that we are giving more weightage to the features which are more likely to result in survival.
        # For example, if the passenger is a female (sex_female=1), embarked from Cherbourg (embarked_C=1), travelling in first class (class_First=1), 
        # and is an adult male (adult_male_False=1), then the chances of survival are high.
        # Similarly, if the passenger is a male (sex_male=1), embarked from Southampton (embarked_S=1), travelling in third class (class_Third=1), 
        # and is a child (who_child=1), then the chances of survival are low.
        # The fare is also considered, assuming that passengers who paid a higher fare are more likely to survive.
        # The age is considered inversely, assuming that younger passengers are more likely to survive.
        # The sibsp (number of siblings/spouses aboard) and parch (number of parents/children aboard) are also considered inversely, 
        # assuming that passengers with fewer family members aboard are more likely to survive.
        
        y = 0.1*row['sex_female'] + 0.1*row['embarked_C'] + 0.1*row['class_First'] + 0.1*row['adult_male_False'] + 0.05*row['fare'] - 0.05*row['age'] - 0.05*row['sibsp'] - 0.05*row['parch'] - 0.1*row['sex_male'] - 0.1*row['embarked_S'] - 0.1*row['class_Third'] - 0.1*row['who_child']
        
        # The output is then normalized to be between 0 and 1 using the sigmoid function.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)