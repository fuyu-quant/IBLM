import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sentiment score based on the number of positive and negative words in the text
        positive_words = ['good', 'great', 'excellent', 'best', 'amazing', 'love', 'loved', 'like', 'liked', 'enjoy', 'enjoyed', 'favorite', 'wonderful', 'superb', 'fantastic', 'happy', 'strong', 'beautiful', 'awesome', 'perfect', 'recommend', 'recommended']
        negative_words = ['bad', 'worst', 'terrible', 'awful', 'poor', 'disappoint', 'disappointed', 'disappointing', 'hate', 'hated', 'dislike', 'disliked', 'boring', 'weak', 'ugly', 'annoy', 'annoying', 'waste', 'stupid', 'ridiculous', 'lame', 'forgettable']

        text = row['text'].lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()

        positive_count = sum([1 for word in words if word in positive_words])
        negative_count = sum([1 for word in words if word in negative_words])

        sentiment_score = positive_count - negative_count

        # Normalize the sentiment score to a probability value between 0 and 1
        y = 1 / (1 + np.exp(-sentiment_score))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)