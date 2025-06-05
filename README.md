### EX6 Information Retrieval Using Vector Space Model in Python

#### DATE: 

#### AIM: 
To implement Information Retrieval Using Vector Space Model in Python.

#### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

#### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

#### Program:
```
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from tabulate import tabulate

# Sample documents
documents = {
    "doc1": "The cat sat on the mat.",
    "doc2": "The dog sat on the log.",
    "doc3": "The cat lay on the rug.",
}

# Preprocessing without NLTK
def preprocess_text(text):
    tokens = re.findall(r'\b\w+\b', text.lower())  # tokenize and lowercase
    tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS and token not in string.punctuation]
    return " ".join(tokens)

# Preprocess all documents
preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

# Vectorizers
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(preprocessed_docs.values())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

terms = tfidf_vectorizer.get_feature_names_out()

# Term Frequency Table
print("\n--- Term Frequencies (TF) ---\n")
tf_table = count_matrix.toarray()
print(tabulate([["Doc ID"] + list(terms)] + [[list(preprocessed_docs.keys())[i]] + list(row) for i, row in enumerate(tf_table)], headers="firstrow", tablefmt="grid"))

# Document Frequency (DF) and IDF Table
df = np.sum(count_matrix.toarray() > 0, axis=0)
idf = tfidf_vectorizer.idf_

df_idf_table = []
for i, term in enumerate(terms):
    df_idf_table.append([term, df[i], round(idf[i], 4)])

print("\n--- Document Frequency (DF) and Inverse Document Frequency (IDF) ---\n")
print(tabulate(df_idf_table, headers=["Term", "Document Frequency (DF)", "Inverse Document Frequency (IDF)"], tablefmt="grid"))

# TF-IDF Table
print("\n--- TF-IDF Weights ---\n")
tfidf_table = tfidf_matrix.toarray()
print(tabulate([["Doc ID"] + list(terms)] + [[list(preprocessed_docs.keys())[i]] + list(map(lambda x: round(x, 4), row)) for i, row in enumerate(tfidf_table)], headers="firstrow", tablefmt="grid"))

# Manual Cosine Similarity
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0.0
    return dot_product, norm_vec1, norm_vec2, similarity

# Search Function
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query]).toarray()[0]
    results = []

    for idx, doc_vector in enumerate(tfidf_matrix.toarray()):
        doc_id = list(preprocessed_docs.keys())[idx]
        doc_text = documents[doc_id]
        dot, norm_q, norm_d, sim = cosine_similarity_manual(query_vector, doc_vector)
        results.append([doc_id, doc_text, round(dot, 4), round(norm_q, 4), round(norm_d, 4), round(sim, 4)])

    results.sort(key=lambda x: x[5], reverse=True)
    return results, query_vector

# User Query
query = input("\nEnter your query: ")

# Perform Search
results_table, query_vector = search(query, tfidf_matrix, tfidf_vectorizer)

# Cosine Similarity Table
print("\n--- Search Results and Cosine Similarity ---\n")
headers = ["Doc ID", "Document", "Dot Product", "Query Magnitude", "Doc Magnitude", "Cosine Similarity"]
print(tabulate(results_table, headers=headers, tablefmt="grid"))

# Ranking Table
print("\n--- Ranked Documents ---\n")
ranked_docs = []
for idx, res in enumerate(results_table, start=1):
    ranked_docs.append([idx, res[0], res[1], res[5]])

print(tabulate(ranked_docs, headers=["Rank", "Document ID", "Document Text", "Cosine Similarity"], tablefmt="grid"))

# Highest Score
highest_score = max(row[5] for row in results_table)
print(f"\nThe highest rank cosine score is: {highest_score}")

# Query TF-IDF Weights
print("\n--- Query TF-IDF Weights ---\n")
query_weights = [(terms[i], round(query_vector[i], 4)) for i in range(len(terms)) if query_vector[i] > 0]
print(tabulate(query_weights, headers=["Term", "Query TF-IDF Weight"], tablefmt="grid"))
```
#### Output:

![{B6AC129E-9540-4998-8AB0-5314B2A240D4}](https://github.com/user-attachments/assets/c6f58991-37b7-4191-bfc7-b3449034977d)
![{6C70FA3F-7BF0-47AF-A687-DF569E35FE43}](https://github.com/user-attachments/assets/78ffb33e-c42e-442d-b3b0-79435a27488c)
![{517F7BC7-DF97-4F81-838E-DAB75FC93F0E}](https://github.com/user-attachments/assets/6a4e488c-dd55-472b-adc2-a3db02a2e5a9)

#### Result:
Thus the, Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, TF-IDF scores, and performing similarity calculations between queries and documents is executed successfully.
