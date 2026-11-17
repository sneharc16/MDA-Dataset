import re
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def preprocess(text, verbose=False):
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())   # lowercase, no punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    doc = nlp(cleaned)
    tokens = [t.text for t in doc]                     # tokenisation
    no_stop = [t for t in doc if not t.is_stop]        # stopword removal
    lemmas = [t.lemma_ for t in no_stop]               # lemmatisation
    if verbose:
        print("Original    :", text)
        print("Cleaned     :", cleaned)
        print("Tokens      :", tokens)
        print("No stopwords:", [t.text for t in no_stop])
        print("Lemmatised  :", lemmas, "\n")
    return " ".join(lemmas)

# ---------- Part A: single review (reviews_1_entry.csv) ----------
df1 = pd.read_csv("reviews_1_entry.csv")
print("reviews_1_entry.csv:", df1.shape)
processed = preprocess(df1.loc[0, "ReviewText"], verbose=True)

tfidf_1 = TfidfVectorizer()
X1 = tfidf_1.fit_transform([processed])
print("Vocabulary  :", list(tfidf_1.get_feature_names_out()))
print("TF-IDF shape:", X1.shape)
print("TF-IDF row  :", X1.toarray().round(4), "\n")

# ---------- Part B: five reviews (reviews_5_entries) ----------
df = pd.read_csv("reviews_5_entries.csv")       # same data as reviews_5_entries.pdf
print("reviews_5_entries:", df.shape)
df["Processed"] = df["ReviewText"].apply(preprocess)
print(df[["ReviewID", "Processed"]].to_string(index=False), "\n")

bow = CountVectorizer()
X_bow = bow.fit_transform(df["Processed"])
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df["Processed"])
vocab = tfidf.get_feature_names_out()

print("Vocabulary size:", len(vocab))
print("Vocabulary     :", list(vocab), "\n")
print("BoW matrix shape   :", X_bow.shape)
print("TF-IDF matrix shape:", X_tfidf.shape, "\n")

tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vocab,
                        index=[f"R{i}" for i in df.ReviewID])
print("TF-IDF vector of review 1 (non-zero terms):")
print(tfidf_df.loc["R1"][tfidf_df.loc["R1"] > 0].round(4).to_string(), "\n")

for rid, row in tfidf_df.iterrows():
    top = row.sort_values(ascending=False).head(3)
    print(f"{rid} top terms:", ", ".join(f"{w} ({v:.2f})" for w, v in top.items()))

# Heatmap of the TF-IDF matrix
plt.figure(figsize=(11, 3.2))
plt.imshow(tfidf_df.values, cmap="Blues", aspect="auto")
plt.colorbar(label="TF-IDF weight")
plt.xticks(range(len(vocab)), vocab, rotation=60, ha="right", fontsize=8)
plt.yticks(range(len(tfidf_df)), tfidf_df.index)
plt.title("TF-IDF matrix: 5 reviews x vocabulary")
plt.tight_layout()
plt.savefig("exp1_tfidf_heatmap.png", dpi=200)
