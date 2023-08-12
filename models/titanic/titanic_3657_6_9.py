Here is a simple example of a prediction function. This function calculates the mean of each column for the target 0 and 1, and then calculates the Euclidean distance from the unknown data to the mean of each target. The function returns the probability of the target being 1, which is calculated as the inverse of the distance to the mean of target 1 divided by the sum of the inverses of the distances to the means of both targets.

```python
import numpy as np
import pandas as pd
from scipy.spatial import distance

def predict(x):
    df = x.copy()
    output = []
    target0_mean = df[df['target'] == 0].mean()
    target1_mean = df[df['target'] == 1].mean()
    for index, row in df.iterrows():
        dist_to_target0 = distance.euclidean(row[:-1], target0_mean)
        dist_to_target1 = distance.euclidean(row[:-1], target1_mean)
        prob_target1 = 1 / dist_to_target1 / (1 / dist_to_target0 + 1 / dist_to_target1)
        output.append(prob_target1)
    return np.array(output)
```

Please note that this is a very simple and naive approach and may not give accurate results for complex datasets. For more accurate predictions, you should consider using machine learning models such as logistic regression, decision trees, or neural networks.