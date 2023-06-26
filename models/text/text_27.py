import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'astonishing', 'amazing', 'wonderful', 'great', 'good', 'love', 'best', 'favorite', 'recommend', 'enjoy', 'beautiful', 'charming', 'happy', 'fun', 'interesting', 'moving', 'insightful', 'riveting', 'captivating', 'powerful', 'strong', 'superb', 'brilliant', 'talented', 'impressive', 'incredible', 'fantastic', 'memorable', 'touching', 'heartfelt', 'emotional', 'engaging', 'entertaining', 'well-done', 'well-crafted', 'well-acted', 'well-written', 'well-directed', 'masterpiece', 'classic', 'must-see', 'highly']
        negative_words = ['awful', 'terrible', 'horrible', 'bad', 'worst', 'disappointing', 'boring', 'dull', 'uninteresting', 'forgettable', 'waste', 'lame', 'stupid', 'ridiculous', 'poor', 'weak', 'uninspired', 'unoriginal', 'predictable', 'cliché', 'pointless', 'lousy', 'crappy', 'garbage', 'trash', 'disaster', 'flop', 'failure', 'mediocre', 'unconvincing', 'unbelievable', 'unimpressive', 'annoying', 'irritating', 'painful', 'tiresome', 'tedious', 'dragging', 'flat', 'unappealing', 'unengaging', 'unentertaining', 'poorly-done', 'poorly-crafted', 'poorly-acted', 'poorly-written', 'poorly-directed', 'avoid']

        positive_count = sum([1 for word in positive_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)])
        negative_count = sum([1 for word in negative_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)])

        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count + 1e-6)
        y = (sentiment_score + 1) / 2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)