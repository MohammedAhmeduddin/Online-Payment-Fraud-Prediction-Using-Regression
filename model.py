#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #library for data analysis
import numpy as np #library used for working with arrays.


# In[2]:


df=pd.read_csv("/Users/ahmedbinnayeem/Desktop/Fraud_Prediction/OnlineFraudPrediction/online_payment.csv")

# In[6]:

# The function dataframe. isnull(). sum(). sum() returns the number of missing values in the data set.


# ### Data Correlation

# In[9]:


df.corr()
del df["nameOrig"]
del df["nameDest"]

# The corr() method finds the correlation of each column in a DataFrame.


# # Data Visualization

# In[4]:


#import matplotlib.pyplot as plt
#plt.scatter(df["oldbalanceOrg"],df["newbalanceOrig"])
#plt.title("oldbalanceOrg vs newbalanceOrig")
#plt.xlabel("oldbalanceOrg")
#plt.ylabel("newbalanceOrig")


# In[6]:


#plt.scatter(df["amount"],df["isFlaggedFraud"])
#plt.title("amount vs isFlaggedFraud")
#plt.xlabel("amount")
#plt.ylabel("isFlaggedFraud")


# In[5]:


#import seaborn as sns
#sns.heatmap(df.corr(), annot = True)


# In[13]:


#sns.pairplot(df)
#plt.show()


# In[15]:


#sns.violinplot( x = 'isFraud', y = 'isFlaggedFraud', data = df)
#plt.show()


# In[16]:


#sns.stripplot( x = 'type', y = 'amount', data = df)
#plt.show()


# In[18]:


#sns.violinplot( x = 'type', y = 'isFraud', data = df)
#plt.show()


# In[7]:


#sns.countplot(df['isFlaggedFraud'])
#plt.show()


# In[8]:


#sns.countplot(df['isFraud'])
#plt.show()


# In[10]:


from sklearn.preprocessing import LabelEncoder

# LabelEncoder can be used to normalize labels. 
# It can also be used to transform non-numerical labels to numerical labels.


# In[11]:


le=LabelEncoder()


# In[12]:


var=df.select_dtypes(include="object").columns


# In[14]:


for i in var:
    df[i]=le.fit_transform(df[i])


# # Under Sampling Method 

# In[15]:

# =============================================================================
# 
# # class count
#count_class_0, count_class_1 = df.isFraud.value_counts()
# 
# #Divide by class
#df_class_0 = df[df['isFraud']==0]
#df_class_1 = df[df['isFraud']==1]
# 
# 
# # In[16]:
# 
# 
#count_class_0, count_class_1 
# 
# 
# # In[17]:
# 
# 
# df_class_0.shape
# 
# 
# # In[18]:
# 
# 
# df_class_1.shape
# 
# 
# # In[19]:
# 
# 
# df_class_0.sample(count_class_1).shape
# 
# 
# # In[20]:
# 
# 
#df_class_0_under = df_class_0.sample(count_class_1)
# 
# # concatenation of class 0 df and class 1 df
#df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
# 
# 
# # In[21]:
# 
# 
# df_test_under.shape
# 
# 
# # In[22]:
# 
# 
# print('Random under-sampling:')  # verifying both the class have same no. of samples
# print(df_test_under.isFraud.value_counts())
# 
# 
# # # Train and Test Split
# 
# # In[23]:
# 
# 
#x= df_test_under.drop('isFraud', axis= 'columns')
#y= df_test_under['isFraud']

x=df.drop('isFraud', axis=1)
y=df['isFraud']

# 
# =============================================================================
from sklearn.model_selection import train_test_split
x_train, x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2, random_state=15, stratify=y)

#stratify : which will make sure you have balance samples
# the samples in x train and x test will have balanced samples from 0 and 1 class stratify
# will help to ensure this principle 

# # Training Model using Decision Tree Classifier

# In[27]:


from sklearn.tree import DecisionTreeClassifier


# In[30]:


model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[31]:


print("Training accuracy: ", model.score(x_train, y_train)*100)


# In[33]:


print("Testing accuracy: ", model.score(x_test, y_test)*100)


# In[ ]:

import pickle
pickle.dump(model,open('model.pkl','wb'))


