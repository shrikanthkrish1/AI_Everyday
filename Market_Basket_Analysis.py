#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data1=pd.read_csv('Groceries data.csv')
data2=pd.read_csv('basket.csv')


# In[3]:


data1.head()


# In[18]:


data1.info()


# In[19]:


data1.isnull().sum()


# In[4]:


data2.head()


# In[23]:


data2.info()


# In[24]:


data2.isnull().sum()


# In[5]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# In[6]:


data2


# In[7]:


data2.fillna('1',inplace=True)
transactions=[]
for i in range(14963):
    transaction=[]
    for j in range(11):
        if data2.iloc[i,j]!='1':
            transaction.append(data2.iloc[i,j])
    transactions.append(transaction)


# In[10]:


te=TransactionEncoder()
te_bin=te.fit_transform(transactions)
Transactions=pd.DataFrame(te_bin,columns = te.columns_)


# In[11]:


Transactions


# In[12]:


def encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
    
Transactions=Transactions.applymap(encode)


# In[13]:


Transactions


# In[15]:


frequent_items = apriori(Transactions, min_support = 0.004,use_colnames = True)
frequent_items.head()


# In[16]:


rules = association_rules(frequent_items, metric='lift',min_threshold =1)
rules.head()


# In[17]:


rules = rules.sort_values(by='lift', ascending = False)
rules


# In[ ]:




