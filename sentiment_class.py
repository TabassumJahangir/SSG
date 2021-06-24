#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle

vectorizer = pickle.load(open('pickles/vectorizer.sav', 'rb'))
classifier = pickle.load(open('pickles/classifier.sav', 'rb'))

text = 'Your service is worst Im not going to buy anything from your company again'
text_vector = vectorizer.transform([text])
result = classifier.predict(text_vector)
print(result)

