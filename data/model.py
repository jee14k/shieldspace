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







"""
Full‑stack message analyser
───────────────────────────
• Emotions                 (SamLowe/roberta‑base‑go_emotions)
• General toxicity         (unitary/toxic‑bert)
• Sexism / Racism detector (jkos0012/sexism_racism_bert_model   – weights in .pth)
"""


# !pip install -qU transformers huggingface_hub torch safetensors nltk

# ── 1. Imports ───────────────────────────────────────────────────
import textwrap, nltk, torch
from huggingface_hub import hf_hub_download
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

# NLTK data (only if you really need punkt/vader later)
# nltk.download("punkt")
# nltk.download("vader_lexicon")

# ── 2. Emotion & general‑toxicity pipelines (unchanged) ─────────
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

# ── 3. Sexism / Racism detector built from raw .pth  ────────────
REPO_ID      = "jkos0012/sexism_racism_bert_model"
PT_FILE      = "sexism_racism_bert_model.pth"
TOKENIZER_ID = "bert-base-uncased"      
LABELS       = ["sexism", "racism"]     

# 3‑A  download the .pth checkpoint once and load it
ckpt_path  = hf_hub_download(REPO_ID, filename=PT_FILE)
state_dict = torch.load(ckpt_path, map_location="cpu")

# 3‑B  build a compatible BERT config with num_labels=2
config = AutoConfig.from_pretrained(
    TOKENIZER_ID,
    num_labels=len(LABELS),
    id2label=dict(enumerate(LABELS)),
    label2id={l: i for i, l in enumerate(LABELS)},
)

# 3‑C  recreate the classification model and load weights
bias_model = AutoModelForSequenceClassification.from_config(config)
missing, unexpected = bias_model.load_state_dict(state_dict, strict=False)
assert not missing,  f"Missing weights! {missing}"
assert not unexpected,f"Unexpected keys! {unexpected}"

# 3‑D  wrap in a normal pipeline (sigmoid = multi‑label)
bias_classifier = TextClassificationPipeline(
    model=bias_model,
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_ID),
    function_to_apply="sigmoid",
    top_k=None,
)

# ── 4. Helper functions ─────────────────────────────────────────
def get_top5_emotions(text: str) -> dict:
    res = emotion_classifier(text)[0]
    res.sort(key=lambda x: x["score"], reverse=True)
    return {r["label"]: r["score"] for r in res[:5]}

def get_toxicity_scores(text: str) -> dict:
    return {r["label"]: r["score"] for r in toxicity_classifier(text)[0]}

def get_bias_scores(text: str) -> dict:
    return {r["label"]: r["score"] for r in bias_classifier(text)[0]}

# ── 5. Display logic  ───────────────────────────────────────────
def display_analysis(message: str) -> None:
    emotions = get_top5_emotions(message)
    toxicity = get_toxicity_scores(message)
    bias     = get_bias_scores(message)

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

    print("\n💡 Emotion Analysis (Top 5)")
    for lbl, sc in emotions.items():
        print(f"  {lbl:<14}{sc:6.3f}")
    print(f"🧠 Primary Emotion: {primary_emotion}")

    print("\n⚠️  General Toxicity")
    print(f"  Level: {friendly}")
    print(f"  Tags : {tag_str}")

    print("\n🚨  Sexism / Racism Scores")
    for lbl, sc in bias.items():
        print(f"  {lbl:<6}{sc:6.3f}")

    print("─" * 80 + "\n")

# ── 6. Simple CLI loop ─────────────────────────────────────────
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