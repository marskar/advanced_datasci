
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Import Libraries and Data:

# In[14]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import datetime as dt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# Input data files are available in the "../input/" directory.

# Any results I write to the current directory are saved as output.

# In[15]:


#load training file
train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
print(train.head())
print('---------------------')
print(train.shape)


# Distribution of Target Variable:

# In[16]:


log_errors = train['logerror']
upper_lim = np.percentile(log_errors, 99.5)
lower_lim = np.percentile(log_errors, 0.5)
log_errors = log_errors.clip(lower=lower_lim, upper=upper_lim)


# In[17]:


plt.figure(figsize=(12,10))
plt.hist(log_errors, bins=300)
plt.title('Distribution of Target Variable (log-error)')
plt.ylabel('count')
plt.xlabel('log-error')
plt.show()


# Log-errors are close to normally distributed around a 0 mean, but with a slightly positive skew. There are also a considerable number of outliers, I will explore whether removing these improves model performance.
# 
# Proportion of Missing Values in Each Column:

# In[18]:


#load property features/description file
prop = pd.read_csv("../input/properties_2016.csv")
print(prop.head())
print('---------------------')
print(prop.shape)


# In[19]:


nans = prop.drop('parcelid', axis=1).isnull().sum()
nans.sort_values(ascending=True, inplace=True)
nans = nans / prop.shape[0]
#print(nans)


# In[20]:


plt.figure(figsize=(14, 5))
plt.bar(range(len(nans.index)), nans.values)
plt.xticks(range(len(nans.index)), nans.index.values, rotation=90)
plt.show()


# There are several columns which have a very high proportion of missing values. It may be worth analysing these more closely.
# 
# Monthly Effects on Target Variable

# In[21]:


train['transaction_month'] = pd.DatetimeIndex(train['transactiondate']).month
train.sort_values('transaction_month', axis=0, ascending=True, inplace=True)
print(train.head())

ax = sns.stripplot(x=train['transaction_month'], y=train['logerror'])


# For submission we are required to predict values for October, November and December. The differing distributions of the target variable over these months indicates that it may be useful to create an additional 'transaction_month' feature as shown above. Lets have a closer look at the distribution across only October, November and December.

# In[22]:


ax1 = sns.stripplot(x=train['transaction_month'][train['transaction_month'] > 9], y=train['logerror'])


# Proportion of Transactions in Each Month

# In[23]:


trans = train['transaction_month'].value_counts(normalize=True)
trans = pd.DataFrame(trans)
trans['month'] = trans.index
trans = trans.sort_values('month', ascending=True)
trans.set_index('month')
trans.rename({'transaction_month' : ''})
print(trans)

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(12, 5))
plt.bar(range(len(months)), trans['transaction_month'])
plt.title('Proportion of Transactions per Month')
plt.ylabel('Proportion')
plt.xlabel('Month')
plt.xticks(range(len(months)), months, rotation=90)
plt.show()


# This datase contains more transactions occuring in the Spring and Summer months, although it must be noted that some transactions from October, November and December have been removed to form the competition's test set (thanks to nonrandom for pointing this out).
# 
# Feature Importance

# In[24]:


#fill NaN values with -1 and encode object columns 
for x in prop.columns:
    prop[x] = prop[x].fillna(-1)

#many more parcelids in properties file, merge with training file
train = pd.merge(train, prop, on='parcelid', how='left')
print(train.head())
print('---------------------')
print(train.shape)


# In[25]:


for c in train[['transactiondate', 'hashottuborspa', 'propertycountylandusecode', 'propertyzoningdesc', 'fireplaceflag', 'taxdelinquencyflag']]:
    label = LabelEncoder()
    label.fit(list(train[c].values))
    train[c] = label.transform(list(train[c].values))

x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train['logerror']


# In[26]:


print(x_train.head())
print('------------')
print(y_train.head())


# In[27]:


rf = RandomForestRegressor(n_estimators=30, max_features=None)

rf.fit(x_train, y_train)

rf_importance = rf.feature_importances_


importance = pd.DataFrame()
importance['features'] = x_train.columns
importance['importance'] = rf_importance
print(importance.head())


# In[28]:


importance.sort_values('importance', axis=0, inplace=True, ascending=False)

print('------------')
print(importance.head())


# In[29]:


fig = plt.figure(figsize=(10, 4), dpi=100)
plt.bar(range(len(importance)), importance['importance'])
plt.title('Feature Importances')
plt.xlabel('Feature Name')
plt.ylabel('Importance')
plt.xticks(range(len(importance)), importance['features'], rotation=90)
plt.show()


# Here we see that the greatest importance in predicting the log-error comes from features involving taxes and geographical location of the property. Notably, the 'transaction_month' feature that was engineered earlier was the 12th most important feature. 

# In[30]:


test= test.rename(columns={'ParcelId': 'parcelid'}) 
#To make it easier for merging datasets on same column_id later


# # Analyse the Dimensions of our Datasets

# In[ ]:


print("Training Size:" + str(train.shape))
print("Property Size:" + str(prop.shape))
print("Sample Size:" + str(test.shape))
