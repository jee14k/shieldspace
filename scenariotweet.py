from flask import Flask, jsonify
import pandas as pd
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer

scenariotweet = Flask(__name__)

# Load your dataset
df = pd.read_csv('./cleaned/twitter_cleaned_dataset.csv')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

@scenariotweet.route('/random-tweet-sentiment', methods=['GET'])
def random_tweet_sentiment():
    # Select a random tweet from the dataset
    random_tweet = df['cleaned_text'].sample(1).iloc[0]
    
    # Get the sentiment score for the tweet
    sentiment_score = sia.polarity_scores(random_tweet)
    
    # Return the result as JSON
    return jsonify({
        'tweet': random_tweet,
        'sentiment_score': sentiment_score
    })

if __name__ == '__main__':
    scenariotweet.run(debug=True)
