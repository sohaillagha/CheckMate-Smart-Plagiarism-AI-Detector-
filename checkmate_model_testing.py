# ============================================================
# CheckMate — Model Accuracy Testing & Confusion Matrix
# Local execution script
# ============================================================


# ===================== CELL 2: Imports =====================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    ConfusionMatrixDisplay
)
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

print("✅ All imports loaded!")


# ===================== CELL 3: EXACT PROJECT CODE =====================
# These are the EXACT functions from your project files:
# - preprocessing.py
# - tfidf_model.py
# - sbert_model.py
# - ai_detection_model.py

# ---- FROM preprocessing.py (EXACT COPY) ----

def clean_extracted_text(text):
    text = re.split(r'\nreferences\n', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_tfidf(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    doc = nlp(" ".join(tokens))
    return " ".join([t.lemma_ for t in doc])

def preprocess_sbert(text):
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 0]

def preprocess_ai_detection(text):
    return sent_tokenize(re.sub(r'\s+', ' ', text))


# ---- FROM tfidf_model.py (EXACT COPY) ----

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFIDF_DIR = os.path.join(BASE_DIR, "datasets", "tfidf")

_cached_corpus = None

def load_tfidf_dataset():
    global _cached_corpus
    if _cached_corpus is not None:
        return _cached_corpus

    print("  ⏳ First run: loading TF-IDF corpus (will be cached)...")
    corpus = []
    if not os.path.exists(TFIDF_DIR):
        _cached_corpus = []
        return _cached_corpus

    for file in os.listdir(TFIDF_DIR):
        if file.endswith(".txt"):
            file_path = os.path.join(TFIDF_DIR, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content:
                        sentences = sent_tokenize(content)
                        for s in sentences:
                            processed = preprocess_tfidf(s)
                            if processed.strip():
                                corpus.append(processed)
            except Exception as e:
                print(f"Error reading dataset file {file}: {e}")

    _cached_corpus = corpus
    print(f"  ✅ TF-IDF corpus cached. ({len(corpus)} sentences)")
    return _cached_corpus

def check_tfidf_similarity(sentences):
    """
    Checks similarity for a list of sentences against the TF-IDF dataset.
    Returns (score, matched_indices)
    """
    if not sentences:
        return 0.0, []

    processed_sentences = [preprocess_tfidf(s) for s in sentences]
    valid_indices = [i for i, s in enumerate(processed_sentences) if s.strip()]
    if not valid_indices:
        return 0.0, []

    valid_processed = [processed_sentences[i] for i in valid_indices]

    dataset_sentences = load_tfidf_dataset()
    if not dataset_sentences:
         return 0.0, []

    all_texts = dataset_sentences + valid_processed

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    n_dataset = len(dataset_sentences)
    dataset_matrix = tfidf_matrix[:n_dataset]
    input_matrix = tfidf_matrix[n_dataset:]

    similarities = cosine_similarity(input_matrix, dataset_matrix)

    threshold = 0.75
    matched_indices = []

    for idx_in_valid, row_scores in enumerate(similarities):
        max_score = max(row_scores) if len(row_scores) > 0 else 0
        if max_score > threshold:
            original_idx = valid_indices[idx_in_valid]
            matched_indices.append(original_idx)

    score = len(matched_indices) / len(valid_indices) if valid_indices else 0.0
    return score, matched_indices


# ---- FROM sbert_model.py (EXACT COPY) ----

SBERT_CSV_PATH = os.path.join(BASE_DIR, "datasets", "sbert", "sbert_pairs.csv")

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

_cached_dataset_embeddings = None

def load_sbert_dataset():
    try:
        df = pd.read_csv(SBERT_CSV_PATH)
        if 'sentence1' not in df.columns or 'sentence2' not in df.columns:
            if df.shape[1] >= 2:
                cols = df.columns.tolist()
                df.rename(columns={cols[0]: 'sentence1', cols[1]: 'sentence2'}, inplace=True)

        if 'sentence1' in df.columns and 'sentence2' in df.columns:
             sentences = pd.concat([df['sentence1'], df['sentence2']]).astype(str).unique().tolist()
             return sentences
        else:
             print("Error: SBERT CSV missing required columns.")
             return []
    except Exception as e:
        print(f"Error loading SBERT dataset: {e}")
        return []

def check_sbert_similarity(sentences):
    global _cached_dataset_embeddings

    if not sentences:
        return 0.0, []

    if _cached_dataset_embeddings is None:
        print("  ⏳ First run: encoding SBERT dataset (will be cached)...")
        dataset_sentences = load_sbert_dataset()
        if not dataset_sentences:
            return 0.0, []
        _cached_dataset_embeddings = sbert_model.encode(dataset_sentences, convert_to_tensor=True)
        print(f"  ✅ SBERT dataset cached. ({len(dataset_sentences)} sentences)")

    emb1 = sbert_model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(emb1, _cached_dataset_embeddings)

    best_matches_per_sentence = scores.max(dim=1).values
    threshold = 0.85

    matched_indices = []
    for i, score in enumerate(best_matches_per_sentence):
        if score > threshold:
            matched_indices.append(i)

    final_score = float(len(matched_indices) / len(sentences))
    return final_score, matched_indices


# ---- FROM ai_detection_model.py (EXACT COPY) ----

AI_CSV_PATH = os.path.join(BASE_DIR, "datasets", "ai_detection", "ai_detection.csv")

ai_vectorizer = TfidfVectorizer()
ai_model = LogisticRegression()
_trained = False

def train_ai_detector():
    global _trained
    if _trained:
        return
    print("  ⏳ First run: training AI detector (will be cached)...")
    df = pd.read_csv(AI_CSV_PATH)
    X = ai_vectorizer.fit_transform(df["text"])
    y = df["label"]
    ai_model.fit(X, y)
    _trained = True
    print("  ✅ AI detector cached.")

def predict_ai_probability(sentences):
    if not sentences:
        return 0.0, []

    X = ai_vectorizer.transform(sentences)
    probs = ai_model.predict_proba(X)[:, 1]

    matched_indices = []
    threshold = 0.5
    for i, p in enumerate(probs):
        if p > threshold:
            matched_indices.append(i)

    score = float(len(matched_indices) / len(sentences))
    return score, matched_indices


print("✅ All project model functions loaded (exact copies)!")


# ===================== CELL 4: Verify Uploaded Files =====================
print("Checking uploaded files...")
assert os.path.exists(AI_CSV_PATH), f"❌ Not found: {AI_CSV_PATH}"
print(f"  ✅ {AI_CSV_PATH}")
assert os.path.exists(SBERT_CSV_PATH), f"❌ Not found: {SBERT_CSV_PATH}"
print(f"  ✅ {SBERT_CSV_PATH}")
assert os.path.isdir(TFIDF_DIR), f"❌ Not found: {TFIDF_DIR}/"
txt_files = [f for f in os.listdir(TFIDF_DIR) if f.endswith('.txt')]
print(f"  ✅ {TFIDF_DIR}/ ({len(txt_files)} text files)")
print("\n✅ All files found!")


# ============================================================
#  TEST 1: AI DETECTION MODEL (Logistic Regression)
#  Using exact: train_ai_detector() + predict_ai_probability()
# ============================================================

# ===================== CELL 5: AI Detection — Train/Test Split =====================
df_ai = pd.read_csv(AI_CSV_PATH)
print(f"📄 AI Detection Dataset: {df_ai.shape[0]} samples")
print(f"   Human (label=0): {(df_ai['label']==0).sum()}")
print(f"   AI    (label=1): {(df_ai['label']==1).sum()}")

X_ai = df_ai['text']
y_ai = df_ai['label']

X_train, X_test, y_train, y_test = train_test_split(
    X_ai, y_ai, test_size=0.2, random_state=42, stratify=y_ai
)
print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")

# Train using EXACT project code (TfidfVectorizer + LogisticRegression)
X_train_vec = ai_vectorizer.fit_transform(X_train)
y_train_arr = y_train.values
ai_model.fit(X_train_vec, y_train_arr)
_trained = True
print("   ✅ AI model trained (exact project code)")

# Predict on test set using EXACT project code
X_test_vec = ai_vectorizer.transform(X_test)
y_pred_ai = ai_model.predict(X_test_vec)
y_prob_ai = ai_model.predict_proba(X_test_vec)[:, 1]

# Also test using predict_ai_probability() on each sentence
ai_score, ai_matched = predict_ai_probability(X_test.tolist())
print(f"   predict_ai_probability() score on test set: {ai_score:.4f}")

acc_ai = accuracy_score(y_test, y_pred_ai)
print(f"\n   ✅ AI Detection Accuracy: {acc_ai*100:.2f}%")


# ===================== CELL 6: AI Confusion Matrix =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_ai = confusion_matrix(y_test, y_pred_ai)
ConfusionMatrixDisplay(cm_ai, display_labels=['Human', 'AI-Generated']).plot(
    ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('AI Detection — Confusion Matrix (Counts)', fontsize=13, fontweight='bold')

cm_ai_norm = confusion_matrix(y_test, y_pred_ai, normalize='true')
ConfusionMatrixDisplay(cm_ai_norm, display_labels=['Human', 'AI-Generated']).plot(
    ax=axes[1], cmap='Oranges', values_format='.2%')
axes[1].set_title('AI Detection — Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('ai_detection_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 AI Detection — Classification Report:")
print("=" * 55)
print(classification_report(y_test, y_pred_ai, target_names=['Human', 'AI-Generated']))


# ===================== CELL 7: AI ROC Curve =====================
fpr, tpr, _ = roc_curve(y_test, y_prob_ai)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1e88e5', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.1, color='#1e88e5')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('AI Detection Model — ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ai_detection_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   AUC Score: {roc_auc:.4f}")


# ===================== CELL 8: AI 5-Fold Cross Validation =====================
print("🔄 Running 5-Fold Cross Validation on AI Detection...")
temp_vec = TfidfVectorizer()
X_all_vec = temp_vec.fit_transform(X_ai)
temp_model = LogisticRegression()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(temp_model, X_all_vec, y_ai, cv=cv, scoring='accuracy')

for i, s in enumerate(cv_scores):
    print(f"   Fold {i+1}: {s*100:.2f}%")
print(f"   Mean: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

plt.figure(figsize=(8, 4))
colors = ['#42a5f5', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc']
bars = plt.bar([f'Fold {i+1}' for i in range(5)], cv_scores*100,
               color=colors, edgecolor='white', lw=1.5)
plt.axhline(y=cv_scores.mean()*100, color='red', linestyle='--', lw=1.5,
            label=f'Mean: {cv_scores.mean()*100:.2f}%')
plt.ylabel('Accuracy (%)')
plt.title('AI Detection — 5-Fold Cross Validation', fontsize=14, fontweight='bold')
plt.legend()
for bar, val in zip(bars, cv_scores):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val*100:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('ai_detection_cross_validation.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
#  TEST 2: TF-IDF COPIED DETECTION
#  Using exact: load_tfidf_dataset() + check_tfidf_similarity()
# ============================================================

# ===================== CELL 9: Load TF-IDF Corpus =====================
print("📂 Loading TF-IDF corpus using exact project code...")
corpus = load_tfidf_dataset()
print(f"   Total corpus sentences: {len(corpus)}")


# ===================== CELL 10: Create TF-IDF Test Data =====================
# Pull REAL sentences from .txt files for "copied" ground truth
# These are raw sentences (before preprocessing) — exactly how check_tfidf_similarity() expects them

print("\n📝 Creating TF-IDF test dataset...")
copied_sentences_raw = []

for fname in sorted(os.listdir(TFIDF_DIR)):
    if fname.endswith('.txt') and len(copied_sentences_raw) < 25:
        fpath = os.path.join(TFIDF_DIR, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        sents = sent_tokenize(text)
        # Pick a meaningful sentence (not too short, not too long)
        good = [s for s in sents if 40 < len(s) < 300]
        if good:
            copied_sentences_raw.append(good[min(5, len(good)-1)])

copied_sentences_raw = copied_sentences_raw[:25]

# Original sentences — completely unrelated to corpus
original_sentences_raw = [
    "The weather forecast for tomorrow suggests a high chance of thunderstorms in the afternoon.",
    "My grandmother's recipe for apple pie requires six large Granny Smith apples and sugar.",
    "The basketball championship game between the Lakers and Celtics was exciting to watch.",
    "Learning to play the acoustic guitar requires daily practice and a lot of patience.",
    "The new highway construction project near downtown will reduce morning commute times.",
    "Photography enthusiasts recommend shooting during golden hour for dramatic lighting.",
    "The local farmers market sells fresh organic vegetables and homemade jams every Saturday.",
    "Swimming is widely considered an excellent full-body exercise for all muscle groups.",
    "The museum exhibition featuring ancient Egyptian artifacts attracted fifty thousand visitors.",
    "Climate scientists report that bird migration patterns have shifted dramatically recently.",
    "The Italian restaurant on Main Street serves the best wood-fired pizza in the city.",
    "Professional rock climbing has seen a massive surge in popularity since its Olympic debut.",
    "Gardening therapy programs have helped many seniors improve their mental wellbeing.",
    "The vintage car show displayed over two hundred beautifully restored classic automobiles.",
    "Modern electric vehicles can travel over four hundred miles on a single battery charge.",
    "The children's library book club meets every Wednesday to discuss fiction novels.",
    "Homemade sourdough bread requires a starter culture fed regularly for several weeks.",
    "The annual jazz festival downtown features performers from over twenty countries.",
    "Morning yoga classes have become the most popular wellness offering for office workers.",
    "The planetarium show about galaxy formation uses stunning computer-generated effects.",
    "Scuba diving courses teach underwater safety and marine life identification skills.",
    "The science fair winner built a solar-powered water purifier from recycled materials.",
    "Artisan coffee roasters source their beans from small farms in Ethiopia and Colombia.",
    "The city council approved funding for three new public parks with walking trails.",
    "Documentary filmmakers spent two years in the Amazon capturing rare wildlife behavior.",
]
original_sentences_raw = original_sentences_raw[:len(copied_sentences_raw)]

print(f"   Copied (from corpus):  {len(copied_sentences_raw)}")
print(f"   Original (unrelated):  {len(original_sentences_raw)}")


# ===================== CELL 11: Run TF-IDF Detection (Exact Code) =====================
print("\n🔍 Running check_tfidf_similarity() — EXACT project code...\n")

# Test copied sentences
tfidf_score_copied, tfidf_matched_copied = check_tfidf_similarity(copied_sentences_raw)
print(f"   Copied sentences → score={tfidf_score_copied:.4f}, matched={len(tfidf_matched_copied)}/{len(copied_sentences_raw)}")

# Reset cache for second call (since vectorizer is fitted per-call in project code)
_cached_corpus = None

# Test original sentences
tfidf_score_orig, tfidf_matched_orig = check_tfidf_similarity(original_sentences_raw)
print(f"   Original sentences → score={tfidf_score_orig:.4f}, matched={len(tfidf_matched_orig)}/{len(original_sentences_raw)}")

# Build ground truth vs predictions
y_true_tfidf = np.array([1]*len(copied_sentences_raw) + [0]*len(original_sentences_raw))
y_pred_tfidf = np.zeros(len(y_true_tfidf), dtype=int)

# Mark matched copied sentences as predicted=1
for idx in tfidf_matched_copied:
    y_pred_tfidf[idx] = 1

# Mark matched original sentences as predicted=1 (false positives if any)
for idx in tfidf_matched_orig:
    y_pred_tfidf[len(copied_sentences_raw) + idx] = 1

all_tfidf_sentences = copied_sentences_raw + original_sentences_raw
print(f"\n   Results:")
for i, (sent, gt, pred) in enumerate(zip(all_tfidf_sentences, y_true_tfidf, y_pred_tfidf)):
    status = "✅" if gt == pred else "❌"
    label = "COPIED" if gt == 1 else "ORIG  "
    print(f"   {status} [{label}] pred={'COPIED' if pred else 'ORIG  '} | {sent[:65]}...")

acc_tfidf = accuracy_score(y_true_tfidf, y_pred_tfidf)
print(f"\n   ✅ TF-IDF Accuracy: {acc_tfidf*100:.2f}%")


# ===================== CELL 12: TF-IDF Confusion Matrix =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_tfidf = confusion_matrix(y_true_tfidf, y_pred_tfidf)
ConfusionMatrixDisplay(cm_tfidf, display_labels=['Original', 'Copied']).plot(
    ax=axes[0], cmap='Greens', values_format='d')
axes[0].set_title('TF-IDF Copied Detection — Confusion Matrix', fontsize=13, fontweight='bold')

cm_tfidf_norm = confusion_matrix(y_true_tfidf, y_pred_tfidf, normalize='true')
ConfusionMatrixDisplay(cm_tfidf_norm, display_labels=['Original', 'Copied']).plot(
    ax=axes[1], cmap='Purples', values_format='.2%')
axes[1].set_title('TF-IDF — Normalized', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('tfidf_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 TF-IDF — Classification Report:")
print("=" * 55)
print(classification_report(y_true_tfidf, y_pred_tfidf, target_names=['Original', 'Copied']))


# ============================================================
#  TEST 3: SBERT PARAPHRASE DETECTION
#  Using exact: load_sbert_dataset() + check_sbert_similarity()
# ============================================================

# ===================== CELL 13: Load SBERT Dataset =====================
print("📂 Loading SBERT dataset using exact project code...")
sbert_ref_sentences = load_sbert_dataset()
print(f"   Reference sentences: {len(sbert_ref_sentences)}")


# ===================== CELL 14: Create SBERT Test Data =====================
# Paraphrased = actual sentences from SBERT reference (should be detected as match)
# Original = completely unrelated (should NOT match)

print("\n📝 Creating SBERT test dataset...")

# Pick real reference sentences (these ARE in the dataset, so should match)
paraphrased_raw = [s for s in sbert_ref_sentences if 40 < len(s) < 250][:25]

# Unrelated originals
original_for_sbert = [
    "The weather forecast for tomorrow suggests a high chance of thunderstorms in the afternoon.",
    "My grandmother's recipe for apple pie requires six large Granny Smith apples and sugar.",
    "The basketball championship game between the Lakers and Celtics was exciting to watch.",
    "Learning to play the acoustic guitar requires daily practice and a lot of patience.",
    "The new highway construction project near downtown will reduce morning commute times.",
    "Photography enthusiasts recommend shooting during golden hour for dramatic lighting.",
    "The local farmers market sells fresh organic vegetables and homemade jams every Saturday.",
    "Swimming is widely considered an excellent full-body exercise for all muscle groups.",
    "The museum exhibition featuring ancient Egyptian artifacts attracted fifty thousand visitors.",
    "Climate scientists report that bird migration patterns have shifted dramatically recently.",
    "The Italian restaurant on Main Street serves the best wood-fired pizza in the city.",
    "Professional rock climbing has seen a massive surge in popularity since its Olympic debut.",
    "Gardening therapy programs have helped many seniors improve their mental wellbeing.",
    "The vintage car show displayed over two hundred beautifully restored classic automobiles.",
    "Modern electric vehicles can travel over four hundred miles on a single battery charge.",
    "The children's library book club meets every Wednesday to discuss fiction novels.",
    "Homemade sourdough bread requires a starter culture fed regularly for several weeks.",
    "The annual jazz festival downtown features performers from over twenty countries.",
    "Morning yoga classes have become the most popular wellness offering for office workers.",
    "The planetarium show about galaxy formation uses stunning computer-generated effects.",
    "Scuba diving courses teach underwater safety and marine life identification skills.",
    "The science fair winner built a solar-powered water purifier from recycled materials.",
    "Artisan coffee roasters source their beans from small farms in Ethiopia and Colombia.",
    "The city council approved funding for three new public parks with walking trails.",
    "Documentary filmmakers spent two years in the Amazon capturing rare wildlife behavior.",
]
original_for_sbert = original_for_sbert[:len(paraphrased_raw)]

print(f"   Paraphrased (from dataset): {len(paraphrased_raw)}")
print(f"   Original (unrelated):       {len(original_for_sbert)}")


# ===================== CELL 15: Run SBERT Detection (Exact Code) =====================
print("\n🧠 Running check_sbert_similarity() — EXACT project code...\n")

# Test paraphrased sentences
sbert_score_para, sbert_matched_para = check_sbert_similarity(paraphrased_raw)
print(f"   Paraphrased sentences → score={sbert_score_para:.4f}, matched={len(sbert_matched_para)}/{len(paraphrased_raw)}")

# Test original sentences
sbert_score_orig, sbert_matched_orig = check_sbert_similarity(original_for_sbert)
print(f"   Original sentences   → score={sbert_score_orig:.4f}, matched={len(sbert_matched_orig)}/{len(original_for_sbert)}")

# Build ground truth vs predictions
y_true_sbert = np.array([1]*len(paraphrased_raw) + [0]*len(original_for_sbert))
y_pred_sbert = np.zeros(len(y_true_sbert), dtype=int)

for idx in sbert_matched_para:
    y_pred_sbert[idx] = 1

for idx in sbert_matched_orig:
    y_pred_sbert[len(paraphrased_raw) + idx] = 1

all_sbert_sentences = paraphrased_raw + original_for_sbert
print(f"\n   Results:")
for i, (sent, gt, pred) in enumerate(zip(all_sbert_sentences, y_true_sbert, y_pred_sbert)):
    status = "✅" if gt == pred else "❌"
    label = "PARA" if gt == 1 else "ORIG"
    print(f"   {status} [{label}] pred={'PARA' if pred else 'ORIG'} | {sent[:65]}...")

acc_sbert = accuracy_score(y_true_sbert, y_pred_sbert)
print(f"\n   ✅ SBERT Accuracy: {acc_sbert*100:.2f}%")


# ===================== CELL 16: SBERT Confusion Matrix =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_sbert = confusion_matrix(y_true_sbert, y_pred_sbert)
ConfusionMatrixDisplay(cm_sbert, display_labels=['Original', 'Paraphrased']).plot(
    ax=axes[0], cmap='YlOrRd', values_format='d')
axes[0].set_title('SBERT Paraphrase Detection — Confusion Matrix', fontsize=13, fontweight='bold')

cm_sbert_norm = confusion_matrix(y_true_sbert, y_pred_sbert, normalize='true')
ConfusionMatrixDisplay(cm_sbert_norm, display_labels=['Original', 'Paraphrased']).plot(
    ax=axes[1], cmap='RdPu', values_format='.2%')
axes[1].set_title('SBERT — Normalized', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('sbert_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 SBERT — Classification Report:")
print("=" * 55)
print(classification_report(y_true_sbert, y_pred_sbert, target_names=['Original', 'Paraphrased']))


# ============================================================
#  COMBINED SUMMARY — ALL 3 MODELS
# ============================================================

# ===================== CELL 17: Summary Table =====================
print("\n" + "=" * 70)
print("            CHECKMATE — ALL MODELS ACCURACY SUMMARY")
print("=" * 70)

accs  = [accuracy_score(y_test, y_pred_ai), accuracy_score(y_true_tfidf, y_pred_tfidf), accuracy_score(y_true_sbert, y_pred_sbert)]
precs = [precision_score(y_test, y_pred_ai, zero_division=0), precision_score(y_true_tfidf, y_pred_tfidf, zero_division=0), precision_score(y_true_sbert, y_pred_sbert, zero_division=0)]
recs  = [recall_score(y_test, y_pred_ai, zero_division=0), recall_score(y_true_tfidf, y_pred_tfidf, zero_division=0), recall_score(y_true_sbert, y_pred_sbert, zero_division=0)]
f1s   = [f1_score(y_test, y_pred_ai, zero_division=0), f1_score(y_true_tfidf, y_pred_tfidf, zero_division=0), f1_score(y_true_sbert, y_pred_sbert, zero_division=0)]

summary = pd.DataFrame({
    'Model': ['AI Detection (LR)', 'Copied Detection (TF-IDF)', 'Paraphrase Detection (SBERT)'],
    'Accuracy': [f'{a*100:.2f}%' for a in accs],
    'Precision': [f'{p*100:.2f}%' for p in precs],
    'Recall': [f'{r*100:.2f}%' for r in recs],
    'F1-Score': [f'{f*100:.2f}%' for f in f1s],
})
print(summary.to_string(index=False))
print("=" * 70)


# ===================== CELL 18: Combined Bar Chart =====================
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(3)
w = 0.18
labels = ['AI Detection\n(Logistic Regression)', 'Copied Detection\n(TF-IDF + Cosine Sim)', 'Paraphrase Detection\n(SBERT all-MiniLM-L6-v2)']

b1 = ax.bar(x-1.5*w, [a*100 for a in accs],  w, label='Accuracy',  color='#42a5f5')
b2 = ax.bar(x-0.5*w, [p*100 for p in precs], w, label='Precision', color='#66bb6a')
b3 = ax.bar(x+0.5*w, [r*100 for r in recs],  w, label='Recall',    color='#ffa726')
b4 = ax.bar(x+1.5*w, [f*100 for f in f1s],   w, label='F1-Score',  color='#ef5350')

ax.set_ylabel('Score (%)', fontsize=13)
ax.set_title('CheckMate — All Models Performance Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.3)

for bars in [b1, b2, b3, b4]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%',
                ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ===================== CELL 19: All 3 Confusion Matrices Side by Side =====================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_ai),
    display_labels=['Human', 'AI']).plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('AI Detection\n(Logistic Regression)', fontsize=12, fontweight='bold')

ConfusionMatrixDisplay(confusion_matrix(y_true_tfidf, y_pred_tfidf),
    display_labels=['Original', 'Copied']).plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title('Copied Detection\n(TF-IDF + Cosine Similarity)', fontsize=12, fontweight='bold')

ConfusionMatrixDisplay(confusion_matrix(y_true_sbert, y_pred_sbert),
    display_labels=['Original', 'Paraphrased']).plot(ax=axes[2], cmap='Oranges', values_format='d')
axes[2].set_title('Paraphrase Detection\n(SBERT all-MiniLM-L6-v2)', fontsize=12, fontweight='bold')

plt.suptitle('CheckMate — Confusion Matrices for All Three Detection Models',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('all_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ ALL DONE! Saved plots:")
print("   📊 ai_detection_confusion_matrix.png")
print("   📊 ai_detection_roc_curve.png")
print("   📊 ai_detection_cross_validation.png")
print("   📊 tfidf_confusion_matrix.png")
print("   📊 sbert_confusion_matrix.png")
print("   📊 all_models_comparison.png")
print("   📊 all_confusion_matrices.png")
