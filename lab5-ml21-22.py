#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# In[23]:


import os
for dirname, _, filenames in os.walk(r'A:\MTECH(Data Science)\DataSet\Machin learing Lab\5\lab5.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[24]:


#import
data = pd.read_csv(r'A:\MTECH(Data Science)\DataSet\Machin learing Lab\5\lab5.csv')


# In[25]:


#The total number of training instances are :
data


# In[26]:


X = data.iloc[:,:-1]
print("\nThe First 5 values of train data is\n",X.head())
y = data.iloc[:,-1]
print("\nThe first 5 values of Train output is\n",y.head())


# In[27]:


le_outlook = LabelEncoder()
X.Outlook = le_outlook.fit_transform(X.Outlook)
le_Temperature = LabelEncoder()
X.Temperature = le_Temperature.fit_transform(X.Temperature)
le_Humidity = LabelEncoder()
X.Humidity = le_Humidity.fit_transform(X.Humidity)
le_Wind = LabelEncoder()
X.Wind = le_Wind.fit_transform(X.Wind)


# In[28]:


print("\nNow the Train data is :\n",X.head())
le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is\n",y)


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
classifier = GaussianNB()
classifier.fit(X_train,y_train)


# In[20]:


from sklearn.metrics import accuracy_score
print("Accuracy is:",accuracy_score(classifier.predict(X_test),y_test))

