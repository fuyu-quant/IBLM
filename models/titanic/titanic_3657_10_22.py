Here is a simple example of a prediction function that uses a basic rule-based approach. This function assumes that if a passenger is female, they have a higher chance of survival (target=1), and if they are male, they have a lower chance of survival (target=0). This is based on the historical fact that women and children were given priority to lifeboats on the Titanic.

```python
import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # If the passenger is female, predict a high probability of survival
        if row['sex_female'] == 1.0:
            y = 0.9
        # If the passenger is male, predict a low probability of survival
        elif row['sex_male'] == 1.0:
            y = 0.1
        # If the gender is unknown, predict a neutral probability of survival
        else:
            y = 0.5
        output.append(y)
    return np.array(output)
```

This is a very simple model and does not take into account many other factors that could influence survival, such as age, class, or fare. A more complex model could be created by adding more conditions and rules based on these other factors. However, creating a highly accurate model without using machine learning techniques would be very difficult and time-consuming.