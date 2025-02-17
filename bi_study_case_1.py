import json
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from prophet import Prophet

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
datesList = []

for review_list in reviews:
    for review in review_list:
        text = review['text']['text']
        text = text.replace('\n\n', ' ')
        text = text.replace('\n', ' ')
        reviewsList.append(text)
        ratingsList.append(review['rating'])
        datesList.append(review['publishTime'])

data = {
    'review': reviewsList,
    'rating': ratingsList,
    'date': datesList
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


# Predictive analysis of sentiment scores
def categorize_sentiment(score):
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'


df['sentiment'] = df['sentiment_score'].apply(categorize_sentiment)
df['date'] = pd.to_datetime(df['date'])
daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
daily_sentiment['cap'] = 1
daily_sentiment['floor'] = -1

prophet_df = daily_sentiment.rename(columns={'date': 'ds', 'sentiment_score': 'y'})
model = Prophet(growth='logistic', weekly_seasonality=False, daily_seasonality=True)
model.fit(prophet_df)
future = model.make_future_dataframe(periods=120)
future['cap'] = 1
future['floor'] = -1

forecast = model.predict(future)
fig1 = model.plot(forecast)
ax = fig1.gca()
ax.set_xlabel('Date')
ax.set_ylabel('Sentiment Score')
ax.set_title('Future Sentiment Score Prediction')
plt.show()
