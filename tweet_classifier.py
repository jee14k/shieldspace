from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
#model = joblib.load('analysis/sexism_racism_model.pkl')
#vectorizer = joblib.load('analysis/tfidf_vectorizer.pkl')

# Load the trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load('analysis/sexism_racism_bert_model.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

neutral_keywords = ['hi', 'hello', 'good morning', 'test', 'ok', 'lol', 'hey', 'nice day', 'bye', 'thanks', 'thank you', 'goodbye', 'see you', 'take care', 'have a nice day', 'good night', 'good luck', 'happy birthday', 'congratulations', 'welcome', 'cheers', 'best wishes', 'peace', 'love', 'smile', 'friend', 'friendship', 'family', 'fun', 'happy', 'joy', 'laughter', 'enjoy', 'relax', 'chill', 'vacation', 'holiday', 'travel', 'adventure', 'explore', 'discover', 'learn', 'knowledge', 'wisdom', 'growth', 'improvement', 'success', 'achievement', 'goal', 'dream', 'aspiration', 'motivation', 'inspiration', 'creativity', 'art', 'music', 'dance', 'theater', 'film', 'literature', 'poetry', 'writing', 'storytelling', 'imagination', 'fantasy', 'science fiction', 'mystery', 'thriller', 'comedy', 'drama']

def is_likely_neutral(tweet):
    tweet_lower = tweet.lower().strip()
    return tweet_lower in neutral_keywords or len(tweet_lower.split()) <= 2

# Function to predict whether the tweet is sexist or racist
def predict_tweet(tweet):
    if is_likely_neutral(tweet):
        return "This tweet is likely neutral "


    # Tokenize the input tweet
    inputs = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        max_length=128,  # You can adjust this length if needed
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move tensors to the same device as the model
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get prediction from the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get predicted class
    prediction = torch.argmax(logits, dim=1).item()

    # Return result based on prediction
    return "Sexist" if prediction == 1 else "Racist"

# Define the main route
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        tweet = request.form['tweet']
        result = predict_tweet(tweet)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)