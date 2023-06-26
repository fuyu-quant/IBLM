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
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    df['text'] = df['text'].apply(preprocess_text)

    # Calculate word frequencies
    word_freq = Counter()
    for text in df['text']:
        word_freq.update(text.split())

    # Calculate sentiment scores
    def sentiment_score(text):
        words = text.split()
        pos_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'favorite', 'enjoy', 'happy']
        neg_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'boring', 'disappointing', 'sad']
        pos_count = sum([word_freq[word] for word in words if word in pos_words])
        neg_count = sum([word_freq[word] for word in words if word in neg_words])
        return (pos_count - neg_count) / (pos_count + neg_count + 1)

    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        score = sentiment_score(row['text'])
        y = 1 / (1 + np.exp(-score))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)