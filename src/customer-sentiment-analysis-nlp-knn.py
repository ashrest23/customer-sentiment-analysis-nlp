"""
Customer Sentiment Analysis using TF-IDF + Custom KNN

Pipeline Overview
-----------------
1. Load training dataset
2. Perform advanced text preprocessing
      • lowercase
      • punctuation removal
      • stopword removal
      • stemming
3. Convert text into TF-IDF vectors
      • unigram + bigram features
      • L2 normalization
4. Perform k-fold cross validation to select best K
5. Train final custom KNN classifier
6. Generate predictions for test dataset
7. Save predictions to output file

Model
-----
Manual implementation of K-Nearest Neighbors using cosine similarity.

Dataset
-------
Customer sentiment dataset containing text reviews and labels.

Author: Anish Shrestha
"""

############################################################
# IMPORT LIBRARIES
############################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import string

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


############################################################
# TEXT PREPROCESSING
############################################################

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Perform advanced text cleaning.

    Steps
    -----
    1. Convert text to lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove stopwords
    5. Apply stemming

    Parameters
    ----------
    text : str

    Returns
    -------
    cleaned_text : str
    """

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(f"[{string.punctuation}]", " ", text)

    # remove numbers
    text = re.sub(r"\d+", "", text)

    tokens = text.split()

    cleaned_tokens = []

    for token in tokens:

        if token not in stop_words:

            stemmed = stemmer.stem(token)

            cleaned_tokens.append(stemmed)

    return " ".join(cleaned_tokens)


def preprocess_texts(text_list):
    """
    Apply cleaning pipeline to list of documents.
    """

    processed = []

    for text in text_list:

        processed.append(clean_text(text))

    return np.array(processed)


############################################################
# CUSTOM KNN IMPLEMENTATION
############################################################

class KNearestNeighbor:
    """
    Manual implementation of KNN classifier using cosine similarity.

    Parameters
    ----------
    k : int
        Number of nearest neighbors.
    """

    def __init__(self, k=5):

        self.k = k

    def fit(self, X_train, y_train):
        """
        Store training vectors and labels.
        """

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict class labels for test samples.
        """

        predictions = []

        for i in range(X_test.shape[0]):

            similarity = cosine_similarity(
                X_test[i],
                self.X_train
            ).flatten()

            nearest_neighbors = similarity.argsort()[-self.k:]

            neighbor_labels = self.y_train[nearest_neighbors]

            majority_vote = Counter(neighbor_labels).most_common(1)[0][0]

            predictions.append(majority_vote)

        return np.array(predictions)


############################################################
# CROSS VALIDATION FOR K SELECTION
############################################################

def cross_validate_knn(X_text, y, k_values):
    """
    Perform stratified k-fold cross validation to determine optimal K.

    Parameters
    ----------
    X_text : array
        List of text documents
    y : array
        Labels
    k_values : list
        Candidate K values

    Returns
    -------
    scores : list
        Mean accuracy for each K
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    for k in k_values:

        fold_scores = []

        for train_index, val_index in skf.split(X_text, y):

            X_train = X_text[train_index]
            X_val = X_text[val_index]

            y_train = y[train_index]
            y_val = y[val_index]

            ################################################
            # TF-IDF Vectorization
            ################################################

            vectorizer = TfidfVectorizer(

                stop_words="english",

                ngram_range=(1, 2),      # Improvement 2

                max_features=20000,

                norm="l2"                # Improvement 3
            )

            X_train_vec = vectorizer.fit_transform(X_train)

            X_val_vec = vectorizer.transform(X_val)

            ################################################
            # Train KNN
            ################################################

            knn = KNearestNeighbor(k)

            knn.fit(X_train_vec, y_train)

            predictions = knn.predict(X_val_vec)

            accuracy = accuracy_score(y_val, predictions)

            fold_scores.append(accuracy)

        scores.append(np.mean(fold_scores))

    return scores


############################################################
# Plot Error and Accuracy Rate
############################################################

def plot_knn_performance(k_values, accuracy_scores):
    """
    Plot Accuracy and Error Rate for different K values.

    This visualization helps determine the optimal number
    of neighbors for the KNN classifier.

    Parameters
    ----------
    k_values : list
        List of K values tested during cross validation.

    accuracy_scores : list
        Corresponding accuracy scores for each K.

    Returns
    -------
    None
    """

    # Convert to numpy arrays
    k_values = np.array(k_values)
    accuracy_scores = np.array(accuracy_scores)

    # Compute error rate
    error_rates = 1 - accuracy_scores

    plt.figure(figsize=(10, 6))

    # Accuracy plot
    plt.plot(
        k_values,
        accuracy_scores,
        marker='o',
        linestyle='-',
        label='Accuracy'
    )

    # Error rate plot
    plt.plot(
        k_values,
        error_rates,
        marker='s',
        linestyle='--',
        label='Error Rate'
    )

    plt.xlabel("K Value")

    plt.ylabel("Score")

    plt.title("KNN Performance vs K Value")

    plt.legend()

    plt.grid(True)

    plt.show()

############################################################
# MAIN PIPELINE
############################################################

def run_pipeline():

    print("Loading dataset...")

    train_data = pd.read_fwf('traindata.txt', delimiter='\t',sep = '\n',names=["sentiment","review"])
    train_data = train_data.dropna().reset_index(drop=True)
    train_data.loc[[221, 3421, 5021, 5386], 'review'] = "+1"
    train_data['sentiment'] = train_data['sentiment'].str.replace('1. ', '+1', regex=False)
    train_data['sentiment'] = train_data['sentiment'].str.replace('- 1','-1', regex=False)

    test_data = pd.read_fwf('testdata.txt', delimiter='\t',sep = '\n',names=["review"])
    test_data = test_data.dropna().reset_index(drop=True)

    X_train = train_data["review"].values

    Y_train = train_data["sentiment"].values
    

    X_test = test_data["review"].values

    ########################################################
    # TEXT CLEANING
    ########################################################

    print("Cleaning text data...")

    X_train_clean = preprocess_texts(X_train)

    X_test_clean = preprocess_texts(X_test)

    ########################################################
    # CROSS VALIDATION
    ########################################################

    print("Running cross validation...")

    k_values = list(range(3, 50, 2))

    scores = cross_validate_knn(
        X_train_clean,
        Y_train,
        k_values
    )
    
    plot_knn_performance(k_values, scores)
    
    best_k = k_values[np.argmax(scores)]

    print("Best K value:", best_k)

    ########################################################
    # TRAIN FINAL MODEL
    ########################################################

    print("Training final model...")

    vectorizer = TfidfVectorizer(

        stop_words="english",

        ngram_range=(1, 2),

        max_features=20000,

        norm="l2"
    )

    X_train_vec = vectorizer.fit_transform(X_train_clean)

    X_test_vec = vectorizer.transform(X_test_clean)

    knn = KNearestNeighbor(best_k)

    knn.fit(X_train_vec, Y_train)

    predictions = knn.predict(X_test_vec)

    ########################################################
    # SAVE OUTPUT
    ########################################################

    np.savetxt("out.txt", predictions, fmt="%s")

    print("Predictions saved to out.txt")


############################################################
# ENTRY POINT
############################################################

if __name__ == "__main__":

    run_pipeline()