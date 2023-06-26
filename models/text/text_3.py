import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'astonishing', 'amazing', 'wonderful', 'great', 'good', 'love', 'best', 'favorite', 'recommend', 'enjoy', 'beautiful', 'charming', 'happy', 'fun', 'interesting', 'memorable', 'insightful', 'riveting', 'captivating', 'powerful', 'moving', 'heart']
        negative_words = ['awful', 'terrible', 'horrible', 'worst', 'boring', 'dull', 'disappointing', 'waste', 'forgettable', 'uninteresting', 'poor', 'bad', 'lame', 'ridiculous', 'stupid', 'annoying', 'painful', 'unpleasant', 'uninspired', 'predictable', 'cliché', 'weak', 'insulting']

        positive_count = sum([1 for word in positive_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)])
        negative_count = sum([1 for word in negative_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)])

        if positive_count + negative_count == 0:
            y = 0.5
        else:
            y = positive_count / (positive_count + negative_count)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)