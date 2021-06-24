#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

# Loading dataset
data = pd.read_csv('data/train.csv')

# Data labelling
def getLabel(score):
    if score < -0.5:
        return 'Highly Negative'
    elif score < 0:
        return 'Sightly Negative'
    elif score == 0:
        return 'Neutral'
    elif score <= 0.5:
        return 'Slightly Positive'
    else:
        return 'Highly Positive'

data['Label'] = data['Polarity'].apply(getLabel)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(sublinear_tf = True,
                             max_df = 0.90,
                             min_df = 2,
                             ngram_range = (1, 2))
train_vectors = vectorizer.fit_transform(data['Text'])

# Linear SVM Classifier
linear_svm = svm.SVC(kernel='linear').fit(train_vectors, data['Label'])

