import torch
from huggingface_hub import hf_hub_download
from transformers import (
    pipeline,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    TextClassificationPipeline,
    AutoConfig,
    AutoModelForSequenceClassification,
)


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

# ── 3. Sexism / Racism detector built from raw .pth  ────────────
REPO_ID = "jkos0012/bert-cyberbullying"   # <— new repo
PT_FILE = "bert_cyberbullying.pth"        # file inside repo
BASE_MODEL = "bert-base-uncased"             # backbone

# 3-A  download & load the checkpoint
ckpt_path = hf_hub_download(REPO_ID, filename=PT_FILE)
state_dict = torch.load(ckpt_path, map_location="cpu")

# 3-B  infer output-layer size (e.g. 2 × 768)
num_labels = state_dict["classifier.weight"].shape[0]

# 3-C  give the two output neurons real names
LABELS = ["age", "religion"]          # index 0 → age, index 1 → religion
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
# <— no more size-mismatch!
cyber_model.load_state_dict(state_dict, strict=True)

# 3-E  wrap in a normal HF pipeline
cyberbullying_classifier = TextClassificationPipeline(
    model=cyber_model,
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL),
    function_to_apply="sigmoid",   # multi-label
    top_k=None,
)

# ----------------------Third model for sexism and racism----------
REPO_ID_2 = "jkos0012/sexism_racism_bert_model"
PT_FILE_2 = "sexism_racism_bert_model.pth"
TOKENIZER_ID = "bert-base-uncased"
LABELS_2 = ["racism", "sexism"]

# 3‑A  download the .pth checkpoint once and load it
ckpt_path_2 = hf_hub_download(REPO_ID_2, filename=PT_FILE_2)
state_dict_2 = torch.load(ckpt_path_2, map_location="cpu")

# 3‑B  build a compatible BERT config with num_labels=2
config_2 = AutoConfig.from_pretrained(
    TOKENIZER_ID,
    num_labels=len(LABELS_2),
    id2label=dict(enumerate(LABELS_2)),
    label2id={l: i for i, l in enumerate(LABELS_2)},
)

# 3‑C  recreate the classification model and load weights
bias_model = AutoModelForSequenceClassification.from_config(config_2)
missing, unexpected = bias_model.load_state_dict(state_dict_2, strict=False)
assert not missing,  f"Missing weights! {missing}"
assert not unexpected, f"Unexpected keys! {unexpected}"

# 3‑D  wrap in a normal pipeline (sigmoid = multi‑label)
bias_classifier = TextClassificationPipeline(
    model=bias_model,
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_ID),
    function_to_apply="sigmoid",
    top_k=None,
)

# --------------------- Helper Functions ---------------------


def get_cyber_scores(text: str) -> dict:
    cyber_scores = {}
    for r in cyberbullying_classifier(text)[0]:
        if r["score"] >= 0.94:
            cyber_scores[r["label"]] = r["score"]
    return cyber_scores


def get_bias_scores(text: str) -> dict:
    bias_scores = {}
    for r in bias_classifier(text)[0]:
        if r["score"] >= 0.94:
            bias_scores[r["label"]] = r["score"]
    return bias_scores


def get_transformer_emotions(text):
    """
    Uses the GoEmotions model to compute emotion scores.
    Returns the top 5 emotions (sorted by score) as a dictionary.
    """
    results = emotion_classifier(text)[0]
    # Sort the results in descending order by score, and take the top 5 entries
    sorted_results = sorted(
        results, key=lambda x: x['score'], reverse=True)[:5]
    return {item['label']: "{:.3f}".format(item['score']) for item in sorted_results}


def get_toxicity_score(text):
    """
    Uses the Toxic-BERT model to compute toxicity scores.
    Returns a dictionary mapping each toxicity label to its score.
    """
    results = toxicity_classifier(text)[0]
    return {item['label']: "{:.3f}".format(item['score']) for item in results}


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

    cyber = get_cyber_scores(message)
    bias = get_bias_scores(message)
    return {
        "toxic_level": toxic_level,
        "toxicity": toxicity,
        "emotions": emotions,
        "trigger_emotion": trigger_emotion,
        "bias": bias | cyber,
    }
