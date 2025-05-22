# %%
#import libraries
from flask import Flask, render_template, request, redirect, url_for
import random
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# %%
app = Flask(__name__,template_folder='templates')

# Load the cleaned tweet dataset and compute sentiment
def load_tweets_with_sentiment(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['cleaned_text'])  # Drop rows with NaN in cleaned_text

    sid = SentimentIntensityAnalyzer()

    # Calculate compound sentiment score
    df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Optional: categorize sentiment
    def categorize(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment_label'] = df['sentiment_score'].apply(categorize)

    return df

tweets_df = load_tweets_with_sentiment("../cleaned/twitter_cleaned_dataset.csv")

# %%
@app.route('/random_tweet')
def show_random_tweet():
    random_tweet = tweets_df.sample(1).iloc[0]
    tweet_text = random_tweet['cleaned_text']
    sentiment = random_tweet['sentiment_label']
    score = random_tweet['sentiment_score']

    return render_template('tweet.html', tweet=tweet_text, sentiment=sentiment, score=score)


# %%
if __name__ == '__main__':
    app.run(debug=True)

# %%



