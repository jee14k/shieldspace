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
