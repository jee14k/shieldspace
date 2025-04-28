from flask import Flask, render_template, request
import json

app = Flask(__name__)

# Load questions from JSON file
with open('questions.json') as f:
    questions = json.load(f)

@app.route('/')
def index():
    return render_template('index_gpt.html', questions=questions)

@app.route('/submit', methods=['POST'])
def submit():
    score = 0
    for question in questions:
        selected = request.form.get(str(question['id']))
        if selected == question['answer']:
            score += 1
    return render_template('result.html', score=score, total=len(questions))

if __name__ == '__main__':
    app.run(debug=True)
