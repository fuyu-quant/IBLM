import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Analyze the text and count the positive and negative words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'favorite', 'enjoy', 'beautiful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'boring', 'ugly', 'disappointing']

        positive_count = sum([row['text'].count(word) for word in positive_words])
        negative_count = sum([row['text'].count(word) for word in negative_words])

        # Calculate the probability of the target being 1 (positive)
        total_count = positive_count + negative_count
        if total_count == 0:
            y = 0.5
        else:
            y = positive_count / total_count

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)