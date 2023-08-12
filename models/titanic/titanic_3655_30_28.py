import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, in first class, and embarked from Cherbourg
        # These conditions are chosen based on the general knowledge about the Titanic disaster
        # Female passengers and first-class passengers were given priority during the evacuation
        # Passengers who embarked from Cherbourg had a higher survival rate according to the data analysis
        # The age of the passenger is also considered, younger passengers are more likely to survive
        # The fare is also considered, passengers who paid more are more likely to survive
        # This is a very basic and naive prediction model, a more accurate model would require machine learning algorithms

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.3
        if row['embarked_C'] == 1.0:
            y += 0.2
        if row['age'] <= 30.0:
            y += 0.1
        if row['fare'] >= 30.0:
            y += 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)