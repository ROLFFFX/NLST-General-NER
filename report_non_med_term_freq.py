import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
nltk.download('stopwords')
nltk.download('punkt')

with open("medical_terms.json", 'r') as json_file:
    medical_terms_data = json.load(json_file)

medical_terms = medical_terms_data['medical_terms']

# load the data, TEXT portion only
df = pd.read_csv("data/lung_cancer_notes.csv")
texts = df["TEXT"]

'''
    Helper function for preprocessing the note
    - lowercase conversion, special characters replacement
    - tokenization (using punkt)
'''
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return tokens

processed_texts = texts.apply(preprocess_text)

'''
    Helper function for classifying non medical terms.
    TODO: the medical_terms is an examplary corpus that should be replaced with medical
    corpus later.
'''
def extract_non_medical_terms(tokens, medical_vocab):
    non_medical = [word for word in tokens if word not in medical_vocab]
    return tokens # dummy return, not classified

non_medical_terms = Counter()
for tokens in tqdm(processed_texts, desc="Processing texts"):
    non_medical_terms.update(extract_non_medical_terms(tokens, medical_terms))

'''
    #################
    Frequency Analysis Based on Given Results Below:
    #################
'''
print(non_medical_terms.most_common(100))
non_medical_freq_df = pd.DataFrame(non_medical_terms.items(), columns=["Term", "Frequency"])
non_medical_freq_df.to_csv("output/report_non_medical_freq.csv", index=False)

'''
    #################
    Step 2: TF-IDF
    #################
'''
processed_texts_joined = [" ".join(tokens) for tokens in processed_texts]
# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_texts_joined)

terms = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1  # sum TF-IDF scores across all documents

tfidf_df = pd.DataFrame({
    'Term': terms,
    'TF-IDF': tfidf_scores
}).sort_values(by="TF-IDF", ascending=False)


non_medical_tfidf = tfidf_df[~tfidf_df['Term'].isin(medical_terms)]
print("Top Non-Medical Terms by TF-IDF:")
print(non_medical_tfidf.head(100))

non_medical_tfidf.to_csv("output/report_non_medical_tfidf.csv", index=False)

'''
    #################
    Step 3: TF-IDF visualization
    #################
'''
top_non_medical = non_medical_tfidf.head(20)
terms = top_non_medical['Term'].values
scores = top_non_medical['TF-IDF'].values
plt.figure(figsize=(12, 8))
plt.barh(terms, scores, color='skyblue')
plt.xlabel('TF-IDF Score')
plt.ylabel('Terms')
plt.title('Top 20 Non-Medical Terms by TF-IDF')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()