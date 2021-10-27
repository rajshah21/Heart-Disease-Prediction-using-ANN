#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


heart = pd.read_csv("heart.csv")


# In[14]:


heart.head()


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


heart.corr()


# In[17]:


sns.heatmap(heart.corr())


# In[18]:


sns.countplot(x="ChestPainType" ,data=heart)


# In[19]:


sns.barplot(x="ChestPainType",y="HeartDisease",data=heart)


# In[20]:


#Categorical data to be converted to numerical data


# In[21]:


heart["ChestPainType"].unique()


# In[22]:


heart["RestingECG"].unique()


# In[23]:


heart["ST_Slope"].unique()


# In[24]:


#checking for any null values in dataset


# In[25]:


col_name = [col for col in heart.columns if heart[col].isnull().any()]


# In[26]:


col_name


# In[27]:


# no Null values present. 


# In[ ]:





# In[28]:


#converting categorical data to numeric data 


# In[29]:


from sklearn.preprocessing import LabelEncoder


# In[30]:


label=LabelEncoder()


# In[31]:


heart['ChestPainType_numeric']=label.fit_transform(heart['ChestPainType'])
heart['ChestPainType_numeric']


# In[32]:


heart['RestingECG_numeric']=label.fit_transform(heart['RestingECG'])
heart['RestingECG_numeric']


# In[33]:


heart['ST_Slope_numeric']=label.fit_transform(heart['ST_Slope'])
heart['ST_Slope_numeric']


# In[34]:


heart.head()


# In[35]:


#droping categorical attributes as we have numeric values.


# In[36]:


heart.drop('ChestPainType', axis=1,inplace=True)


# In[37]:


heart.drop('RestingECG', axis=1,inplace=True)


# In[38]:


heart.drop('ST_Slope', axis=1,inplace=True)


# In[39]:


heart.head()


# In[50]:


heart.Sex=heart.Sex.replace({'M':0,'F':1})


# In[51]:


heart.ExerciseAngina=heart.ExerciseAngina.replace({'N':0,'Y':1})


# In[55]:


X=heart.drop('HeartDisease',axis=1).values


# In[56]:


y=heart['HeartDisease'].values


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[59]:


import tensorflow as tf


# In[60]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[61]:


from tensorflow.keras.layers import Activation


# In[62]:


model = Sequential()

model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')


# In[63]:


model.fit(X_train,y_train, epochs=300, validation_data=(X_test,y_test),verbose=1)


# In[68]:


model_loss=pd.DataFrame(model.history.history)


# In[69]:


model_loss.plot()


# In[70]:


from sklearn.metrics import confusion_matrix,classification_report


# In[75]:


y_pred=model.predict_classes(X_test)


# In[76]:


y_pred


# In[77]:


print(classification_report(y_test,y_pred))


# In[78]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:





# In[79]:


from tensorflow.keras.layers import Dropout


# In[80]:


model = Sequential()

model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')


# In[84]:


model.fit(X_train,y_train, epochs=500, validation_data=(X_test,y_test),verbose=1)


# In[85]:


model_loss_drop=pd.DataFrame(model.history.history)


# In[86]:


model_loss_drop.plot()


# In[87]:


predictions=model.predict_classes(X_test)


# In[88]:


print(classification_report(y_test,predictions))


# In[89]:


print(confusion_matrix(y_test,predictions))


# In[ ]:




