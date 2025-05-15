from flask import Flask, request, render_template_string
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Path to your saved model and tokenizer
model_path = './bert-cyberbullying'

# Load tokenizer and model from local folder
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Label mapping for predictions
label_map = {
    0: "age",
    1: "religion"
}

@app.route('/')
def home():
    return '''
        <h2>Cyberbullying Classification</h2>
        <form method="POST" action="/predict">
            <label>Enter a tweet:</label><br>
            <textarea name="text" rows="4" cols="50" required></textarea><br><br>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']

    # Tokenize input
    encoding = tokenizer(
        input_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

    # Get model prediction
    with torch.no_grad():
        outputs = model(**encoding)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = label_map.get(prediction, "Unknown")

    return f'''
        <h3>Predicted cyberbullying type: <u>{predicted_label}</u></h3>
        <br><a href='/'>Try another</a>
    '''

if __name__ == '__main__':
    app.run(debug=True)
