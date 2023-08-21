#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


penguins = pd.read_csv('penguins_cleaned.csv')
penguins.head()


# In[4]:


df = penguins.copy()
target = 'species'
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
    
df.head()


# In[5]:


target_mapper = { 'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2 }
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)
df.head()


# In[7]:


X = df.drop('species', axis=1)
Y = df['species']
X.head()


# In[12]:


Y.head()


# In[13]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)


# In[14]:


import pickle
with open('penguins_clf.pkl', 'wb') as pkl_file:
    pickle.dump(clf, pkl_file)

