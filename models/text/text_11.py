import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'astonishing', 'amazing', 'wonderful', 'great', 'good', 'love', 'best', 'favorite', 'recommend', 'enjoy', 'beautiful', 'charming', 'happy', 'fun', 'interesting', 'moving', 'insightful', 'riveting', 'captivating', 'powerful', 'memorable', 'heart']
        negative_words = ['awful', 'terrible', 'horrible', 'bad', 'worst', 'hate', 'disappointing', 'boring', 'dull', 'stupid', 'ridiculous', 'lame', 'predictable', 'forgettable', 'waste', 'annoying', 'uninteresting', 'uninspired', 'pointless', 'weak', 'poor', 'painful', 'insulting']

        positive_count = sum([1 for word in positive_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)])
        negative_count = sum([1 for word in negative_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)])

        if positive_count + negative_count == 0:
            y = 0.5
        else:
            y = positive_count / (positive_count + negative_count)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)