#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle

# pickling the vectorizer
pickle.dump(vectorizer, open('pickles/vectorizer.sav', 'wb'))

# pickling the classifier model
pickle.dump(linear_svm, open('pickles/classifier.sav', 'wb'))

