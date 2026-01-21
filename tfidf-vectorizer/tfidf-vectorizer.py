import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    
    if not documents:
        return np.empty((0,)), []

    #tokenization 
    tokenized_docs=[doc.lower().split() if doc.strip() else [] for doc in documents]
    #vocabulary
    vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))
    size_docs = len(tokenized_docs)
    size_vocab = len(vocabulary)

    if size_vocab==0:
        return np.empty((0,)), []

    #tf
    tf=[]
    for doc in tokenized_docs:
        if len(doc)==0:
            tf.append({})
            continue

        docTf = Counter(doc)
        doc_len = len(doc)
        for word in docTf:
            docTf[word] /= doc_len
        tf.append(docTf)

    #idf
    N=size_docs
    i_word ={word:i for i, word in enumerate(vocabulary)}
    

    df_arr = np.zeros(size_vocab)
    for tokens in tokenized_docs:
        unique = set(tokens)
        for term in unique:
            df_arr[i_word[term]] +=1
            
    idf={}
    for word in vocabulary:
        df_val = df_arr[i_word[word]]
        idf[word]=math.log(N/np.maximum(df_val,1))

    #tf-idf
    tfidf_matrix = np.zeros((N,size_vocab))
    for i, docTf in enumerate(tf):
        for word, tfValue in docTf.items():
            j = i_word[word]
            tfidf_matrix[i,j] = tfValue*idf[word]

    return tfidf_matrix,vocabulary