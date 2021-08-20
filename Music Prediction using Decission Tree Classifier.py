#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv('music.csv')
data


# In[5]:


# find missing values inside the data
data.isna().sum()


# In[8]:


# Prepare the data
# i,e sllit tha given data into two parts i,e input and output
# Here 'X'(uppercase x) is the input variable and 'y' (lowercase y) is the output  variable
X = data.drop(columns=['genre'])
y = data['genre']


# In[17]:


# Build a model and make predictions
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier() # building a model
model.fit(X,y) # training a model
predictions = model.predict([[21,1],[32,0],[28,0]]) # making predictions from the model
predictions


# In[ ]:




