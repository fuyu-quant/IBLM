import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given features
        pclass = row['pclass']
        age = row['age']
        fare = row['fare']
        sex_female = row['sex_female']
        sex_male = row['sex_male']
        embarked_C = row['embarked_C']
        embarked_Q = row['embarked_Q']
        embarked_S = row['embarked_S']
        class_First = row['class_First']
        class_Second = row['class_Second']
        class_Third = row['class_Third']
        
        # Weights for each feature
        w_pclass = -0.15
        w_age = -0.005
        w_fare = 0.002
        w_sex_female = 0.5
        w_sex_male = -0.5
        w_embarked_C = 0.1
        w_embarked_Q = 0.05
        w_embarked_S = -0.15
        w_class_First = 0.3
        w_class_Second = 0.1
        w_class_Third = -0.4
        
        # Calculate the probability using a weighted sum of features
        prob = (pclass * w_pclass + age * w_age + fare * w_fare + sex_female * w_sex_female + sex_male * w_sex_male +
                embarked_C * w_embarked_C + embarked_Q * w_embarked_Q + embarked_S * w_embarked_S +
                class_First * w_class_First + class_Second * w_class_Second + class_Third * w_class_Third)
        
        # Normalize the probability to be between 0 and 1
        y = 1 / (1 + np.exp(-prob))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)