import numpy as np
import pandas as pd

def titanic(data):
    # Convert input data to DataFrame
    data = [row.split(',') for row in data.split('\n')]
    columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone', 'target']
    df = pd.DataFrame(data, columns=columns)
    
    # Convert data types
    df['pclass'] = df['pclass'].astype(int)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['sibsp'] = df['sibsp'].astype(int)
    df['parch'] = df['parch'].astype(int)
    df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
    df['target'] = df['target'].astype(int)
    
    # Feature creation and data preprocessing
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['class'] = df['class'].map({'First': 1, 'Second': 2, 'Third': 3})
    df['who'] = df['who'].map({'man': 0, 'woman': 1, 'child': 2})
    df['adult_male'] = df['adult_male'].map({True: 1, False: 0})
    df['alone'] = df['alone'].map({True: 1, False: 0})
    
    # Fill missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df['fare'].fillna(df['fare'].median(), inplace=True)
    
    # Predictive model
    output = []
    for index, row in df.iterrows():
        y = 0
        
        # Weights for features
        y += -0.5 * row['pclass']
        y += 1.5 * row['sex']
        y += -0.02 * row['age']
        y += -0.3 * row['sibsp']
        y += 0.1 * row['parch']
        y += 0.01 * row['fare']
        y += 0.2 * row['embarked']
        y += -0.5 * row['class']
        y += 1.5 * row['who']
        y += -1.5 * row['adult_male']
        y += 0.3 * row['alone']
        
        # Apply logistic function
        y = 1 / (1 + np.exp(-y))
        output.append(y)
    
    output = np.array(output)
    
    return output