Here is a simple example of a prediction function that uses a basic rule-based approach. This function assumes that if a passenger is female, they have a higher probability of survival (target=1), and if they are male, they have a lower probability of survival (target=0). This is based on historical data from the Titanic disaster, where women and children were given priority for lifeboat seats.

```python
import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # If the passenger is female, assign a high probability of survival.
        if row['sex_female'] == 1.0:
            y = 0.9
        # If the passenger is male, assign a low probability of survival.
        elif row['sex_male'] == 1.0:
            y = 0.1
        # If the gender is unknown, assign a neutral probability of survival.
        else:
            y = 0.5
        output.append(y)
    return np.array(output)
```

This is a very simple model and does not take into account many other factors that could influence survival, such as age, class, or fare paid. A more complex model could incorporate these factors to make more accurate predictions. However, without using a machine learning model, creating a complex rule-based model could become very complicated and time-consuming.