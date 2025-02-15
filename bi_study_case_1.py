import json
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Data ingestion and preparation
places = []


def read_json(name):
    with open(name, 'r') as file:
        data = json.load(file)
        places.append(data)


read_json('movati_barheaven.json')
read_json('movati_kanata.json')
read_json('movati_nepean.json')
read_json('movati_orleans.json')
read_json('movati_trainyards.json')


def transform(place):
    return place['reviews']


reviews = list(map(transform, places))
reviewsList = []
ratingsList = []

for review_list in reviews:
    for review in review_list:
        text = review['text']['text']
        text = text.replace('\n\n', ' ')
        text = text.replace('\n', ' ')
        reviewsList.append(text)
        ratingsList.append(review['rating'])

data = {
    'review': reviewsList,
    'rating': ratingsList
}
df = pd.DataFrame(data)

# Sentiment analysis via VADER
sia = SentimentIntensityAnalyzer()


def get_sentiment(review):
    sentiment = sia.polarity_scores(review)
    return sentiment['compound']


df['sentiment_score'] = df['review'].apply(get_sentiment)
print("Data with Sentiment Scores")
print(df)

# Scatter plot visualization of ratings vs VADER sentiment scores
plt.figure(figsize=(8, 5))
plt.scatter(df['rating'], df['sentiment_score'], color='blue')
plt.title('Comparison: Rating vs. VADER Sentiment Score')
plt.xlabel('Rating')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.show()
