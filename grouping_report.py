from sentence_transformers import SentenceTransformer, util
import numpy as np
import tqdm

# Load a pretrained model (e.g., SBERT or BioBERT)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed medical terms
medical_vocab_embeddings = model.encode(list(medical_terms), convert_to_tensor=True)

# Function to classify terms based on similarity
def classify_terms_by_similarity(tokens, medical_vocab_embeddings, model, threshold=0.75):
    non_medical = []
    for word in tokens:
        word_embedding = model.encode(word, convert_to_tensor=True)
        similarity_scores = util.cos_sim(word_embedding, medical_vocab_embeddings).squeeze()
        if torch.max(similarity_scores).item() < threshold:  # If no close match in medical vocab
            non_medical.append(word)
    return non_medical

# Updated logic for processing and classification
non_medical_terms = Counter()
for tokens in tqdm(processed_texts, desc="Processing texts with semantic grouping"):
    non_medical_terms.update(classify_terms_by_similarity(tokens, medical_vocab_embeddings, model))

# Frequency Analysis
print(non_medical_terms.most_common(100))
non_medical_freq_df = pd.DataFrame(non_medical_terms.items(), columns=["Term", "Frequency"])
non_medical_freq_df.to_csv("output/report_non_medical_freq_semantic.csv", index=False)

# TF-IDF analysis (unchanged)
processed_texts_joined = [" ".join(tokens) for tokens in processed_texts]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_texts_joined)

terms = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1

tfidf_df = pd.DataFrame({
    'Term': terms,
    'TF-IDF': tfidf_scores
}).sort_values(by="TF-IDF", ascending=False)

non_medical_tfidf = tfidf_df[~tfidf_df['Term'].isin(medical_terms)]
print("Top Non-Medical Terms by TF-IDF:")
print(non_medical_tfidf.head(100))

non_medical_tfidf.to_csv("output/report_non_medical_tfidf_semantic.csv", index=False)
