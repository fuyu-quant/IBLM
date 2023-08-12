import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, 'pclass' (passenger class) is a significant factor. Passengers in class 1 are more likely to survive than those in classes 2 and 3.
        # Similarly, 'sex_female' is given more weightage as females had a higher survival rate.
        # 'age' is also considered, giving more chances of survival to younger passengers.
        # 'fare' is considered as passengers who paid higher fares are more likely to be in a higher class and thus, have a higher survival rate.
        # 'sibsp' and 'parch' are considered as passengers with siblings/spouses or parents/children on board may have higher survival rates.
        # 'embarked_C' is given more weightage as passengers who embarked at Cherbourg had a higher survival rate.
        # 'alone_True' is considered as passengers travelling alone may have a lower survival rate.
        # 'adult_male_True' is considered as adult males had a lower survival rate.
        # 'class_First' is given more weightage as passengers in the first class had a higher survival rate.
        # 'deck_B', 'deck_D', 'deck_E' are given more weightage as these decks were closer to the lifeboats and thus, passengers in these decks had a higher survival rate.

        y = 0.1*row['pclass'] + 0.3*row['sex_female'] - 0.05*row['age'] + 0.1*row['fare'] + 0.05*row['sibsp'] + 0.05*row['parch'] + 0.1*row['embarked_C'] - 0.05*row['alone_True'] - 0.1*row['adult_male_True'] + 0.1*row['class_First'] + 0.05*row['deck_B'] + 0.05*row['deck_D'] + 0.05*row['deck_E']

        # Normalize the output to a probability between 0 and 1
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)