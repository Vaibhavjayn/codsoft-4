#!/usr/bin/env python
# coding: utf-8

# # TASK: SALES PREDICTION USING PYTHON

# DOMAIN : DATA SCIENCE

# ## Importing the Libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ## Data Collection and Processing

# In[2]:


# Creating a DataFrame using CSV file
data=pd.read_csv("advertising.csv")


# In[3]:


# print the number of rows and columns
data.shape


# In[4]:


# print columns nbames of the DataFrame
data.columns


# In[5]:


# print the DataFrame
data


# In[6]:


# checking the null values in each column of the DataFrame
data.isnull().sum()


# ## Analyzing the Data

# In[7]:


data.describe()


# ## Data Visualisation

# In[8]:


data.hist(bins=30)


# In[9]:


plt.scatter(data["TV"],data["Sales"])
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()

plt.scatter(data["Radio"],data["Sales"])
plt.xlabel("Radio")
plt.ylabel("Sales")
plt.show()

plt.scatter(data["Newspaper"],data["Sales"])
plt.xlabel("Newspaper")
plt.ylabel("Sales")
plt.show()


# In[10]:


sns.heatmap(data.corr(),annot=True)


# ## Splitting the data into Testing and Traning data

# In[11]:


X = data[['TV','Newspaper', 'Radio']]
Y = data['Sales']


# In[12]:


X_train,X_test,Y_train,Y_test=train_test_split(data[["TV"]],data["Sales"],test_size=0.3,random_state=0)


# In[13]:


print(X_train)


# In[14]:


print(Y_train)


# ## Model Training

# In[15]:


model=LinearRegression()
model.fit(X_train,Y_train)


# ## Model Evaluation

# In[16]:


y_predict=model.predict(X_test)
print("Predicted value:\n",y_predict)


# In[17]:


# Caluculating mean absolute error
error=metrics.mean_absolute_error(Y_test,y_predict)
print("Mean absolute error :",error)


# In[18]:


# Caluculating R squared error
r2=metrics.r2_score(Y_test,y_predict)
print(" R squared error :",r2)


# In[19]:


plt.plot(y_predict)


# In[20]:


# plotting a scatter chart
plt.scatter(Y_test,y_predict,c="r")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

