import nltk
from transformers import pipeline
import textwrap

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

# --------------------- Initialize Models ---------------------
# Emotion analysis model using SamLowe's GoEmotions model for improved emotion detection
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,         # Return all scores then we'll pick the top 5
    truncation=True
)

# Toxicity detection model using Toxic-BERT
toxicity_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None,
    truncation=True
)

# --------------------- Helper Functions ---------------------


def get_transformer_emotions(text):
    """
    Uses the GoEmotions model to compute emotion scores.
    Returns the top 5 emotions (sorted by score) as a dictionary.
    """
    results = emotion_classifier(text)[0]
    # Sort the results in descending order by score, and take the top 5 entries
    sorted_results = sorted(
        results, key=lambda x: x['score'], reverse=True)[:5]
    return {item['label']: item['score'] for item in sorted_results}


def get_toxicity_score(text):
    """
    Uses the Toxic-BERT model to compute toxicity scores.
    Returns a dictionary mapping each toxicity label to its score.
    """
    results = toxicity_classifier(text)[0]
    return {item['label']: item['score'] for item in results}


def display_analysis(message):
    """
    Performs and displays both emotion analysis (top 5) and toxicity analysis
    with friendly labels and detected tags.
    """
    # Emotion Analysis
    emotions = get_transformer_emotions(message)
    trigger_emotion = max(emotions, key=emotions.get)

    # Toxicity Analysis
    toxicity = get_toxicity_score(message)
    toxic_level = toxicity.get('toxic', 0.0)
    return {
        "toxic_level": toxic_level,
        "toxicity": toxicity,
        "emotions": emotions,
        "trigger_emotion": trigger_emotion
    }

    # Determine friendly toxicity label based on the toxicity score.
    # if toxic_level > 0.85:
    #     friendly_tox_level = "🔥 Highly Toxic"
    # elif toxic_level > 0.5:
    #     friendly_tox_level = "⚠️ Possibly Offensive"
    # elif toxic_level > 0.2:
    #     friendly_tox_level = "🟡 Mildly Risky"
    # else:
    #     friendly_tox_level = "✅ Low or Safe"

    # Generate descriptive tags if certain toxicity sub-scores exceed thresholds.
    # tags = []
    # if toxicity.get("insult", 0) > 0.6:
    #     tags.append("🔴 Insult")
    # if toxicity.get("obscene", 0) > 0.6:
    #     tags.append("🤬 Obscene")
    # if toxicity.get("severe_toxic", 0) > 0.4:
    #     tags.append("🚨 Severe Toxicity")
    # if toxicity.get("identity_hate", 0) > 0.4:
    #     tags.append("🛑 Identity Hate")
    # if toxicity.get("threat", 0) > 0.3:
    #     tags.append("⚠️ Threat")
    # tags_str = ", ".join(tags) if tags else "No critical flags"

    # Print the analysis results.
    # print("\n" + "-" * 80)
    # print("Message Analyzed:")
    # print(textwrap.fill(message, width=80))

    # print("\n💡 Emotion Analysis (Top 5):")
    # for label, score in emotions.items():
    #     print(f"  {label}: {score:.3f}")
    # print(f"\n🧠 Primary Emotion: {trigger_emotion}")

    # print("\n⚠️ Toxicity Analysis:")
    # print(f"  Toxicity Level: {friendly_tox_level}")
    # print(f"  Detected Tags: {tags_str}")
    # print("-" * 80 + "\n")


# def main():
#     print("Enter a message to analyze its emotions and toxicity.")
#     print("Type 'quit' at any time to exit.\n")

#     while True:
#         message = input("Message: ").strip()
#         if message.lower() == 'quit':
#             print("Exiting. Thank you!")
#             break
#         display_analysis(message)


# if __name__ == "__main__":
#     main()



############################################################ racism detection updated with current model 






#!/usr/bin/env python
"""
Full-stack message analyser
───────────────────────────
• Emotions                 (SamLowe/roberta-base-go_emotions)
• General toxicity         (unitary/toxic-bert)
• Cyber-bullying detector  (jkos0012/bert-cyberbullying – .pth weights)
"""

import textwrap, torch
from huggingface_hub import hf_hub_download
from transformers import (
    pipeline,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    TextClassificationPipeline,
)

# ───────────────────────────────────────────────────────────────
# 1.  Optional: NLTK data (uncomment only if you really need it)
# import nltk
# nltk.download("punkt")
# nltk.download("vader_lexicon")
# ───────────────────────────────────────────────────────────────

# 2.  Emotion & general-toxicity classifiers (unchanged)
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    truncation=True,
)

toxicity_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None,
    truncation=True,
)

# 3.  Cyber-bullying detector built from raw .pth checkpoint
REPO_ID     = "jkos0012/bert-cyberbullying"   # <— new repo
PT_FILE     = "bert_cyberbullying.pth"        # file inside repo
BASE_MODEL  = "bert-base-uncased"             # backbone

# 3-A  download & load the checkpoint
ckpt_path  = hf_hub_download(REPO_ID, filename=PT_FILE)
state_dict = torch.load(ckpt_path, map_location="cpu")

# 3-B  infer output-layer size (e.g. 2 × 768)
num_labels = state_dict["classifier.weight"].shape[0]

# 3-C  give the two output neurons real names
LABELS = ["religion", "age"]          # index 0 → age, index 1 → religion
id2label = dict(enumerate(LABELS))
label2id = {v: k for k, v in id2label.items()}

# 3-D  build a compatible BERT config & model
config = BertConfig.from_pretrained(
    BASE_MODEL,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
cyber_model = BertForSequenceClassification(config)
cyber_model.load_state_dict(state_dict, strict=True)   # <— no more size-mismatch!

# 3-E  wrap in a normal HF pipeline
cyberbullying_classifier = TextClassificationPipeline(
    model=cyber_model,
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL),
    function_to_apply="sigmoid",   # multi-label
    top_k=None,
)

# 4.  Helper functions
def get_top5_emotions(text: str) -> dict:
    res = emotion_classifier(text)[0]
    res.sort(key=lambda x: x["score"], reverse=True)
    return {r["label"]: r["score"] for r in res[:5]}

def get_toxicity_scores(text: str) -> dict:
    return {r["label"]: r["score"] for r in toxicity_classifier(text)[0]}

def get_cyber_scores(text: str) -> dict:
    return {r["label"]: r["score"] for r in cyberbullying_classifier(text)[0]}

# 5.  Pretty console output
def display_analysis(message: str) -> None:
    emotions = get_top5_emotions(message)
    toxicity = get_toxicity_scores(message)
    cyber    = get_cyber_scores(message)

    primary_emotion = max(emotions, key=emotions.get)
    toxic_level     = toxicity.get("toxic", 0.0)

    if   toxic_level > 0.85: friendly = "🔥 Highly Toxic"
    elif toxic_level > 0.50: friendly = "⚠️ Possibly Offensive"
    elif toxic_level > 0.20: friendly = "🟡 Mildly Risky"
    else:                    friendly = "✅ Low or Safe"

    tags = []
    if toxicity.get("insult",        0) > 0.6: tags.append("🔴 Insult")
    if toxicity.get("obscene",       0) > 0.6: tags.append("🤬 Obscene")
    if toxicity.get("severe_toxic",  0) > 0.4: tags.append("🚨 Severe Toxicity")
    if toxicity.get("identity_hate", 0) > 0.4: tags.append("🛑 Identity Hate")
    if toxicity.get("threat",        0) > 0.3: tags.append("⚠️ Threat")
    tag_str = ", ".join(tags) if tags else "No critical flags"

    print("\n" + "─" * 80)
    print(textwrap.fill(message, 80))

    print("\n💡 Emotion Analysis (Top 5)")
    for lbl, sc in emotions.items():
        print(f"  {lbl:<18}{sc:6.3f}")
    print(f"🧠 Primary Emotion: {primary_emotion}")

    print("\n⚠️  General Toxicity")
    print(f"  Level: {friendly}")
    print(f"  Tags : {tag_str}")

    print("\n🚨  Cyber-bullying Scores")
    for lbl, sc in cyber.items():
        print(f"  {lbl:<18}{sc:6.3f}")

    print("─" * 80 + "\n")

# 6.  Simple CLI loop
def main():
    print("Type a message (or 'quit'):\n")
    while True:
        try:
            msg = input("Message: ").strip()
        except (EOFError, KeyboardInterrupt):
            msg = "quit"
        if msg.lower() == "quit":
            print("Exiting. Thank you!")
            break
        display_analysis(msg)

if __name__ == "__main__":
    main()








    ###############################W####################################image detection integrated in the model






# 








# Install required packages
!pip install transformers pillow pytesseract opencv-python --quiet

# Imports
import cv2
import pytesseract
import numpy as np
from PIL import Image
from transformers import pipeline, AutoTokenizer, BertConfig, BertForSequenceClassification, TextClassificationPipeline
import torch
import textwrap
from huggingface_hub import hf_hub_download
from google.colab import files

# Preprocessing + OCR function
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    pil_img = Image.fromarray(enhanced)
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    return text.strip()

# Load HuggingFace models
emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, truncation=True)
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, truncation=True)

REPO_ID = "jkos0012/bert-cyberbullying"
PT_FILE = "bert_cyberbullying.pth"
BASE_MODEL = "bert-base-uncased"

ckpt_path = hf_hub_download(REPO_ID, filename=PT_FILE)
state_dict = torch.load(ckpt_path, map_location="cpu")
num_labels = state_dict["classifier.weight"].shape[0]
LABELS = ["religion", "age"]
id2label = dict(enumerate(LABELS))
label2id = {v: k for k, v in id2label.items()}
config = BertConfig.from_pretrained(BASE_MODEL, num_labels=num_labels, id2label=id2label, label2id=label2id)

cyber_model = BertForSequenceClassification(config)
cyber_model.load_state_dict(state_dict, strict=True)

cyberbullying_classifier = TextClassificationPipeline(
    model=cyber_model,
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL),
    function_to_apply="sigmoid",
    top_k=None,
)

# Analysis Functions
def get_top5_emotions(text):
    res = emotion_classifier(text)[0]
    res.sort(key=lambda x: x["score"], reverse=True)
    return {r["label"]: r["score"] for r in res[:5]}

def get_toxicity_scores(text):
    return {r["label"]: r["score"] for r in toxicity_classifier(text)[0]}

def get_cyber_scores(text):
    return {r["label"]: r["score"] for r in cyberbullying_classifier(text)[0]}

def detect_trigger_words(text, threshold=0.5):
    words = [w for w in text.split() if w.isalpha()]
    toxic_triggers = []
    emotion_triggers = []

    for word in words:
        # Toxicity trigger detection
        toxic_result = toxicity_classifier(word)[0]
        toxic_score = next((r["score"] for r in toxic_result if r["label"] == "toxic"), 0)
        if toxic_score > threshold:
            toxic_triggers.append((word, toxic_score))

        # Emotion trigger detection
        emotion_result = emotion_classifier(word)[0]
        top_emotion = max(emotion_result, key=lambda x: x["score"])
        if top_emotion["score"] > threshold:
            emotion_triggers.append((word, top_emotion["label"], top_emotion["score"]))

    return {
        "toxic_triggers": toxic_triggers,
        "emotion_triggers": emotion_triggers
    }

def display_analysis(message):
    emotions = get_top5_emotions(message)
    toxicity = get_toxicity_scores(message)
    cyber = get_cyber_scores(message)
    primary_emotion = max(emotions, key=emotions.get)
    toxic_level = toxicity.get("toxic", 0.0)

    if toxic_level > 0.85:
        friendly = "🔥 Highly Toxic"
    elif toxic_level > 0.5:
        friendly = "⚠️ Possibly Offensive"
    elif toxic_level > 0.2:
        friendly = "🟡 Mildly Risky"
    else:
        friendly = "✅ Low or Safe"

    tags = []
    if toxicity.get("insult", 0) > 0.6: tags.append("🔴 Insult")
    if toxicity.get("obscene", 0) > 0.6: tags.append("🤬 Obscene")
    if toxicity.get("severe_toxic", 0) > 0.4: tags.append("🚨 Severe Toxicity")
    if toxicity.get("identity_hate", 0) > 0.4: tags.append("🛑 Identity Hate")
    if toxicity.get("threat", 0) > 0.3: tags.append("⚠️ Threat")
    tag_str = ", ".join(tags) if tags else "No critical flags"

    print("\n" + "─" * 80)
    print(textwrap.fill(message, 80))
    print("\n💡 Emotion Analysis (Top 5)")
    for lbl, sc in emotions.items():
        print(f"  {lbl:<18}{sc:6.3f}")
    print(f"🧠 Primary Emotion: {primary_emotion}")
    print("\n⚠️  General Toxicity")
    print(f"  Level: {friendly}")
    print(f"  Tags : {tag_str}")
    print("\n🚨  Cyber-bullying Scores")
    for lbl, sc in cyber.items():
        print(f"  {lbl:<18}{sc:6.3f}")

    # Trigger Word Output
    triggers = detect_trigger_words(message)
    toxic_trigs = triggers["toxic_triggers"]
    emotion_trigs = triggers["emotion_triggers"]

    print("\n🔎 Trigger Words:")
    if toxic_trigs:
        print("  ⚠️ Toxicity Triggers:")
        for word, score in toxic_trigs:
            print(f"    - {word} (score: {score:.2f})")
    else:
        print("  ✅ No strong toxicity trigger words detected.")

    if emotion_trigs:
        print("  💬 Emotional Triggers:")
        for word, label, score in emotion_trigs:
            print(f"    - {word} → {label} (score: {score:.2f})")
    else:
        print("  ✅ No strong emotion trigger words detected.")

    print("─" * 80 + "\n")

# User choice interface
print("What would you like to do?")
print("1️⃣  Upload an image containing text")
print("2️⃣  Enter your own text comment")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    print("📂 Upload your image below...")
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]
    text = extract_text_from_image(image_path)

    if not text:
        print("⚠️ No text found in the image.")
    else:
        print("📄 Extracted Text:\n", text)
        print("\n🔍 Analyzing each line...\n")
        for line in text.split('\n'):
            if line.strip():
                display_analysis(line.strip())

elif choice == "2":
    custom_message = input("✍️ Enter your comment: ").strip()
    if custom_message:
        display_analysis(custom_message)
    else:
        print("⚠️ Empty message entered.")
else:
    print("❌ Invalid choice. Please enter 1 or 2.")