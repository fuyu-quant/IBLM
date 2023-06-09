import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        sex = 1 if row['sex'] == 'male' else 0
        age = row['age']
        pclass = row['pclass']
        fare = row['fare']
        sibsp = row['sibsp']
        parch = row['parch']
        adult_male = 1 if row['adult_male'] else 0
        alone = 1 if row['alone'] else 0

        # Prediction logic
        y = -1.5 + 0.8 * sex - 0.02 * age - 0.5 * pclass + 0.001 * fare - 0.3 * sibsp - 0.2 * parch + 0.6 * adult_male - 0.4 * alone

        y = 1 / (1 + np.exp(-y))
        output.append(y)
    return np.array(output)