#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# To install basic/necessary libraries
get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn')


# In[1]:


# To install basic/necessary libraries
get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn')


# In[6]:


# Import necessarylibraries
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


# In[ ]:





# In[7]:


# Import necessarylibraries
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


# In[8]:


# load the dataset
df = pd.read_csv('PB_All_2000_2021.csv',sep=';')
df


# In[9]:


df.info()


# In[10]:


# rows and cols
df.shape


# In[12]:


# statistics of the data
df.describe().T


# In[13]:


# Missing values

df.isnull().sum()


# In[20]:


# date is in object = date format
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df


# In[21]:


df.info()


# In[22]:


# rows and cols
df.shape


# In[23]:


df = df.sort_values(by=['id','date'])
df.head()


# In[24]:


df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df.head()


# In[26]:


df.columns


# In[27]:


pollutants = [  'O2', 'NO3', 'NO2', 'SO4',
       'PO4', 'CL']


# In[ ]:




