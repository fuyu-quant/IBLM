import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()

    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        A, B, C, D = row['A'], row['B'], row['C'], row['D']

        # Custom logic for binary classification
        y = A * C - B * D

        # Apply sigmoid function to convert y to probability
        y = 1 / (1 + np.exp(-y))
        output.append(y)

    output = np.array(output)
        
    return output

# Example usage:
data = pd.DataFrame({
    'A': [-0.360007291614975, -1.4266828135942111, -0.8713033349256126],
    'B': [-0.3006233204725373, 0.0063185223230947, -0.2553967691046434],
    'C': [1.781525021076077, -1.5914413231764364, 0.9008405013518856],
    'D': [-1.373444579582277, -1.528047075881549, -1.7806373818932837]
})

predictions = predict(data)
print(predictions)