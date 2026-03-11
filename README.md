# Amazon Review Sentiment Classification using k-Nearest Neighbors

## Project Overview

This project implements a **k-Nearest Neighbor (k-NN) classifier** to predict the **sentiment polarity of Amazon product reviews** based on textual content.

The goal is to classify reviews as either:

- **Positive sentiment (+1)**
- **Negative sentiment (-1)**

The classifier is built from scratch without using built-in nearest neighbor implementations. The project focuses on **text preprocessing, feature engineering, similarity measurement, and model selection** to determine the best performing configuration.

This project demonstrates how traditional machine learning algorithms such as k-NN can be applied to **Natural Language Processing (NLP) problems**.

---

## Project Objectives

The main objectives of this project are:

- Implement the **k-Nearest Neighbor classification algorithm from scratch**
- Process and analyze **text-based review data**
- Engineer meaningful **features from raw text**
- Experiment with different **similarity functions and parameters**
- Select the **best performing model configuration**
- Predict sentiment for unseen review data

---

## Dataset

The dataset consists of **Amazon product reviews** used to determine sentiment polarity.

### Training Dataset

- **File:** `traindata.txt`
- **Total reviews:** 18,506
- **Format:**

Each row contains:
<sentiment_label> <review_text>

Example:
+1 This product works perfectly and exceeded my expectations
-1 The quality was terrible and it broke within two days


Where:

- **+1** = Positive sentiment
- **-1** = Negative sentiment

---

### Test Dataset

- **File:** `testdata.txt`
- **Total reviews:** 18,506
- Contains **review text only** (no sentiment labels)

The trained classifier predicts sentiment labels for these reviews.

---

## Machine Learning Approach

### Algorithm

This project implements the **k-Nearest Neighbor (k-NN) classification algorithm**.

The algorithm works as follows:

1. Convert text reviews into numerical feature vectors.
2. Compute similarity between a test review and all training reviews.
3. Identify the **k most similar reviews**.
4. Assign the sentiment label based on the **majority vote** of the nearest neighbors.

---

## Text Preprocessing

To convert raw text into useful features, several preprocessing steps are applied:

- Lowercasing text
- Removing punctuation
- Tokenization
- Removing stopwords (optional)
- Converting text into numerical features

---

## Feature Engineering

Text data must be converted into numeric form for machine learning algorithms.

Possible feature representations explored include:

- **Bag-of-Words**
- **Term Frequency (TF)**
- **TF-IDF**
- Token frequency vectors

Feature engineering plays a key role in improving classification performance.

---

## Similarity Functions

Different similarity or distance metrics can be used to identify nearest neighbors.

Examples include:

- **Cosine Similarity**
- **Euclidean Distance**
- **Manhattan Distance**

Selecting an appropriate similarity function significantly impacts classification accuracy.

---

## Model Selection

Several parameters were evaluated to determine the best model configuration:

| Parameter | Description |
|-----------|-------------|
| k | Number of nearest neighbors |
| Feature representation | Bag-of-Words, TF, TF-IDF |
| Similarity metric | Cosine, Euclidean, Manhattan |

Experiments were conducted to identify the combination of parameters that produced the best results.

---

## Evaluation Metric

Model performance is evaluated using **Accuracy**.

Accuracy measures the proportion of correct predictions:
Accuracy = Correct Predictions / Total Predictions


Predictions generated on the test dataset are compared with hidden ground truth labels to determine leaderboard ranking.

---


This script will:

1. Load and preprocess the training data
2. Convert text reviews into feature vectors
3. Train the k-NN classifier
4. Predict sentiment for the test dataset
5. Output predictions

---

## Output Format

The model outputs sentiment predictions for each test review.

Example:

Each line corresponds to the predicted sentiment of a review in `testdata,txt`.

---

## Technologies Used

- Python
- NumPy
- Natural Language Processing techniques
- Custom implementation of k-Nearest Neighbor algorithm

---

## Key Takeaways

This project demonstrates:

- Implementation of a machine learning algorithm from scratch
- Feature engineering for text-based data
- Application of k-NN to sentiment analysis
- The importance of similarity metrics and parameter tuning

It highlights how classical machine learning methods can be effectively applied to **real-world Natural Language Processing tasks** such as sentiment classification.

---
