import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for passengers who are female, travelling in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where females, first class passengers, and those embarked from Cherbourg had higher survival rates.
        # The age factor is also considered where children are given higher probability.
        # The fare is also considered where passengers who paid higher fares are given higher probability.
        # The conditions are weighted according to their importance.

        y = 0.3*row['sex_female'] + 0.2*(row['pclass'] == 1) + 0.1*(row['embarked_C']) + 0.1*(row['age'] < 18) + 0.3*(row['fare'] > df['fare'].median())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)