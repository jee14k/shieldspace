from flask import Flask, render_template, request, redirect, url_for
import random
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load the cleaned tweet dataset
def load_tweets(file_path):
    df = pd.read_csv(file_path)
    return df['cleaned_text'].dropna().tolist()

#sid = SentimentIntensityAnalyzer()

# Load tweets dataset
tweets = load_tweets("cleaned/cyberbullying_tweets_cleaned.csv")

tweets = pd.DataFrame(tweets)

filtered_df = tweets[
    ~tweets['cyberbullying_type'].isin(['not_cyberbullying', 'other_cyberbullying'])
]

# (optional) Reset the index
#filtered_df = filtered_df.reset_index(drop=True)


#scores = sid.polarity_scores(tweets)

# Routes for different parts of the game
@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/victim', methods=['GET', 'POST'])
def victim_story():
    tweet = random.choice(filtered_df)
    print_message = f"You receive a mean tweet: "
    options = [
        {"label": "A: Ignore", "value": "A"},
        {"label": "B: Respond Kindly", "value": "B"},
        {"label": "C: Report", "value": "C"}
    ]
    if request.method == 'POST':
        choice = request.form['choice']
        if choice == "A":
            result = "You ignore the tweet, but it still affects you emotionally."
        elif choice == "B":
            result = "You respond with kindness, hoping to de-escalate the situation."
        elif choice == "C":
            result = "You report the tweet, and it gets removed. Good decision!"
        else:
            result = "Invalid choice."
        return render_template('scenario_result.html', result=result, next_url='/bystander')
    return render_template('scenario.html', scenario="Victim", tweet=tweet, print_message=print_message, route='/victim', options=options)

@app.route('/bystander', methods=['GET', 'POST'])
def bystander_story():
    tweet = random.choice(filtered_df)
    print_message = f"You see a classmate getting bullied online: "
    options = [
        {"label": "A: Ignore", "value": "A"},
        {"label": "B: Defend the Victim", "value": "B"},
        {"label": "C: Report to an Adult", "value": "C"}
    ]
    if request.method == 'POST':
        choice = request.form['choice']
        if choice == "A":
            result = "You ignore it, but later feel guilty for not stepping in."
        elif choice == "B":
            result = "You stand up for your classmate and tell the bully to stop."
        elif choice == "C":
            result = "You report the incident to a trusted adult, helping to stop the bullying."
        else:
            result = "Invalid choice."
        return render_template('scenario_result.html', result=result, next_url='/aggressor')
    return render_template('scenario.html', scenario="Bystander", tweet=tweet, print_message=print_message, route='/bystander', options=options)

@app.route('/aggressor', methods=['GET', 'POST'])
def aggressor_story():
    tweet = random.choice(filtered_df)
    print_message = f"You post a sarcastic tweet: "
    options = [
        {"label": "A: Apologize", "value": "A"},
        {"label": "B: Justify as a Joke", "value": "B"},
        {"label": "C: Ignore the Reaction", "value": "C"}
    ]
    if request.method == 'POST':
        choice = request.form['choice']
        if choice == "A":
            result = "You apologize and learn that words can hurt, even unintentionally."
        elif choice == "B":
            result = "You justify it as a joke, but the person you targeted is still hurt."
        elif choice == "C":
            result = "You ignore it, but the person stops talking to you."
        else:
            result = "Invalid choice."
        return render_template('scenario_result.html', result=result, next_url='/conclusion')
    return render_template('scenario.html', scenario="Aggressor", tweet=tweet, print_message=print_message, route='/aggressor', options=options)

@app.route('/legal_help')
def legal_help():
    return render_template('legal_help.html')

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

if __name__ == '__main__':
    app.run(debug=True)
