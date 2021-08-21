#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('music.csv')
data


# In[3]:


# find missing values inside the data
data.isna().sum()


# In[4]:


# Prepare the data
# i,e sllit tha given data into two parts i,e input and output
# Here 'X'(uppercase x) is the input variable and 'y' (lowercase y) is the output  variable
X = data.drop(columns=['genre'])
y = data['genre']


# In[25]:


# Build a model and make predictions
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
model = DecisionTreeClassifier() # building a model
model.fit(X_train,y_train) # training a model
predictions = model.predict(X_test) # making predictions from the model
#predictions
score = accuracy_score(y_test, predictions)
score


# In[24]:


#persisting model
import joblib
joblib.dump(model,'music_recommender.joblib')


# In[26]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
#X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
#model = DecisionTreeClassifier() # building a model
#model.fit(X_train,y_train) # training a model
#predictions = model.predict(X_test) # making predictions from the model
#predictions
#score = accuracy_score(y_test, predictions)
#score
model = joblib.load('music_recommender.joblib')
model.predict(X_test)
score


# In[ ]:





# In[ ]:




