import numpy as np
def predict(x):
    df = x.copy()
    df.columns = range(df.shape[1])
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        sex = 1 if row[1] == 'male' else 0
        age = row[2]
        if np.isnan(age):
            age = 30  # median age
        fare = row[5]
        embarked_S = 1 if row[11] == 'Southampton' else 0
        embarked_C = 1 if row[11] == 'Cherbourg' else 0
        embarked_Q = 1 if row[11] == 'Queenstown' else 0
        alone = 1 if row[13] else 0

        # Prediction logic
        y = -1.5 + 0.5 * sex - 0.02 * age + 0.01 * fare + 0.3 * embarked_S - 0.2 * embarked_C - 0.1 * embarked_Q + 0.4 * alone

        y = 1 / (1 + np.exp(-y))
        output.append(y)
    return np.array(output)