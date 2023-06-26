import numpy as np
import re
from collections import Counter

def predict(x):
    df = x.copy()
    output = []

    # Preprocess the text data
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\W+', ' ', text)
        return text

    # Calculate the word frequency in the text
    def word_frequency(text):
        words = text.split()
        word_count = Counter(words)
        return word_count

    # Calculate the sentiment score of the text
    def sentiment_score(text):
        positive_words = ['good', 'great', 'excellent', 'best', 'love', 'amazing', 'awesome', 'wonderful', 'favorite', 'enjoy']
        negative_words = ['bad', 'worst', 'awful', 'terrible', 'hate', 'boring', 'disappointing', 'poor', 'annoying', 'waste']

        word_count = word_frequency(text)
        positive_score = sum(word_count[word] for word in positive_words)
        negative_score = sum(word_count[word] for word in negative_words)

        return (positive_score - negative_score) / (positive_score + negative_score + 1)

    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        preprocessed_text = preprocess_text(text)
        score = sentiment_score(preprocessed_text)

        # Calculate the probability of target being 1
        y = 1 / (1 + np.exp(-score))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)