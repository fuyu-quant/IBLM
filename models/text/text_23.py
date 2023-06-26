import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'astonishing', 'amazing', 'wonderful', 'great', 'good', 'love', 'like', 'enjoy', 'favorite', 'best', 'recommend', 'beautiful', 'happy', 'interesting', 'fun', 'memorable', 'charming', 'superb', 'perfect', 'masterpiece']
        negative_words = ['awful', 'terrible', 'horrible', 'bad', 'hate', 'dislike', 'boring', 'worst', 'disappointing', 'poor', 'lousy', 'ridiculous', 'stupid', 'waste', 'lame', 'forgettable', 'annoying', 'dull', 'predictable', 'uninteresting']

        positive_count = sum([1 for word in positive_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)])
        negative_count = sum([1 for word in negative_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)])

        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count + 1e-6)

        y = (sentiment_score + 1) / 2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)