import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, belongs to first class and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children and first class passengers had higher survival rates.
        # The age factor is also considered where younger passengers are given higher survival probability.
        # The fare is also considered where passengers who paid higher fares are given higher survival probability.
        # This is a simple logic and does not guarantee accurate results for all cases.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.2
        if row['embarked_C'] == 1.0:
            y += 0.1
        if row['age'] <= 18.0:
            y += 0.2
        if row['fare'] >= np.median(df['fare']):
            y += 0.2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)