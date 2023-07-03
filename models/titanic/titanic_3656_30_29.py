import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that we are giving more weightage to the features which are more likely to result in survival.
        # For example, if the passenger is a female (sex_female=1), is in first class (class_First=1), and is an adult (who_woman=1), 
        # then the chances of survival are high. Similarly, if the passenger is a male (sex_male=1), is in third class (class_Third=1), 
        # and is an adult (who_man=1), then the chances of survival are low.
        # We are also considering the age and fare of the passenger. Younger passengers and passengers who paid a higher fare are more likely to survive.
        # The weights for each feature are determined based on their importance in determining the survival of the passenger.

        y = 0.2*row['sex_female'] + 0.15*row['class_First'] + 0.15*row['who_woman'] - 0.2*row['sex_male'] - 0.15*row['class_Third'] - 0.15*row['who_man'] + 0.05*(1 - row['age']/80) + 0.05*(row['fare']/500)

        # The output is a probability value between 0 and 1. We are using the sigmoid function to ensure this.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)