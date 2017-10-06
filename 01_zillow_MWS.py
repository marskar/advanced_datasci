
# coding: utf-8

# In[22]:


from datetime import datetime
d = datetime.now().date()
t = t.strftime('%H:%M:%S')

print("This report was generated on", d, "at", t)


# # Zillow Prize Data Analysis Project

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python (a modified version of this docker image will be made available as part of my project to ensure reproducibility).
# For example, here's several helpful packages to load in 

# Import Libraries and Data:

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import datetime as dt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
get_ipython().run_line_magic('matplotlib', 'inline')
### Seaborn style
sns.set_style("whitegrid")


# Input data files are available in the "../input/" directory.

# Any results I write to the current directory are saved as output.

# In[3]:


## Dictionary of feature dtypes
ints = ['parcelid']

floats = ['basementsqft', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'finishedfloor1squarefeet', 
          'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13',
          'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fireplacecnt',
          'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'latitude', 'longitude',
          'lotsizesquarefeet', 'poolcnt', 'poolsizesum', 'roomcnt', 'threequarterbathnbr', 'unitcnt',
          'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt', 'numberofstories',
          'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear',
          'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyyear']

objects = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
           'buildingqualitytypeid', 'decktypeid', 'fips', 'hashottuborspa', 'heatingorsystemtypeid',
           'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertycountylandusecode',
           'propertylandusetypeid', 'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',
           'regionidcounty', 'regionidneighborhood', 'regionidzip', 'storytypeid',
           'typeconstructiontypeid', 'fireplaceflag', 'taxdelinquencyflag', 'censustractandblock']

feature_dtypes = {col: col_type for type_list, col_type in zip([ints, floats, objects],
                                                               ['int64', 'float64', 'object']) 
                                  for col in type_list}


# In[5]:


### Let's import our data
data = pd.read_csv('./input/properties_2016.csv' , dtype = feature_dtypes)
### and test if everything OK
data.head()


# In[6]:


continuous = ['basementsqft', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 
              'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
              'finishedsquarefeet50', 'finishedsquarefeet6', 'garagetotalsqft', 'latitude',
              'longitude', 'lotsizesquarefeet', 'poolsizesum',  'yardbuildingsqft17',
              'yardbuildingsqft26', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
              'landtaxvaluedollarcnt', 'taxamount']

discrete = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'fireplacecnt', 'fullbathcnt',
            'garagecarcnt', 'poolcnt', 'roomcnt', 'threequarterbathnbr', 'unitcnt',
            'numberofstories', 'assessmentyear', 'taxdelinquencyyear']


# In[10]:


### Continuous variable plots
for col in continuous:
    values = data[col].dropna()
    lower = np.percentile(values, 1)
    upper = np.percentile(values, 99)
    fig = plt.figure(figsize=(18,9));
    sns.distplot(values[(values>lower) & (values<upper)], color='Blue', ax = plt.subplot(121));
    sns.boxplot(y=values, color='Blue', ax = plt.subplot(122));
    plt.suptitle(col, fontsize=16)       


# In[9]:


### Discrete variable plots
NanAsZero = ['fireplacecnt', 'poolcnt', 'threequarterbathnbr']
for col in discrete:
    if col in NanAsZero:
        data[col].fillna(0, inplace=True)
    values = data[col].dropna()   
    fig = plt.figure(figsize=(18,9));
    sns.countplot(x=values, color='Blue', ax = plt.subplot(121));
    sns.boxplot(y=values, color='Blue', ax = plt.subplot(122));
    plt.suptitle(col, fontsize=16)


# In[14]:


### Reading train file
errors = pd.read_csv('./input/train_2016_v2.csv', parse_dates=['transactiondate'])
errors.head()


# In[15]:


#### Merging tables
data_sold = data.merge(errors, how='inner', on='parcelid')
data_sold.head()


# In[16]:


### Creating 5 equal size logerror bins 
data_sold['logerror_bin'] = pd.qcut(data_sold['logerror'], 5, 
                                    labels=['Large Negative Error', 'Medium Negative Error',
                                            'Small Error', 'Medium Positive Error',
                                            'Large Positive Error'])
print(data_sold.logerror_bin.value_counts())


# In[17]:


### Continuous variable vs logerror plots
for col in continuous:     
    fig = plt.figure(figsize=(18,9));
    sns.barplot(x='logerror_bin', y=col, data=data_sold, ax = plt.subplot(121),
                order=['Large Negative Error', 'Medium Negative Error','Small Error',
                       'Medium Positive Error', 'Large Positive Error']);
    plt.xlabel('LogError Bin');
    plt.ylabel('Average {}'.format(col));
    sns.regplot(x='logerror', y=col, data=data_sold, color='Sienna', ax = plt.subplot(122));
    plt.suptitle('LogError vs {}'.format(col), fontsize=16)   


# ## Supplemental figures

# In[22]:


train_df = pd.read_csv("./input/train_2016_v2.csv", parse_dates=["transactiondate"])
train_df.shape


# In[27]:


train_y = train_df['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

from sklearn import ensemble


# In[28]:


import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# In[3]:


plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()


# Distribution of Target Variable:

# In[4]:


log_errors = train['logerror']
upper_lim = np.percentile(log_errors, 99.5)
lower_lim = np.percentile(log_errors, 0.5)
log_errors = log_errors.clip(lower=lower_lim, upper=upper_lim)


# In[5]:


plt.figure(figsize=(12,10))
plt.hist(log_errors, bins=300)
plt.title('Distribution of Target Variable (log-error)')
plt.ylabel('count')
plt.xlabel('log-error')
plt.show()


# Log-errors are close to normally distributed around a 0 mean, but with a slightly positive skew. There are also a considerable number of outliers, I will explore whether removing these improves model performance.
# 
# Proportion of Missing Values in Each Column:

# In[6]:


#load property features/description file
prop = pd.read_csv("../input/properties_2016.csv")
print(prop.head())
print('---------------------')
print(prop.shape)


# # Analyse the Dimensions of our Datasets

# In[21]:


print("Training Size:" + str(train.shape))
print("Property Size:" + str(prop.shape))


# In[7]:


### ... check for NaNs
nan = prop.isnull().sum()
nan


# In[8]:


### Plotting NaN counts
nan_sorted = nan.sort_values(ascending=False).to_frame().reset_index()
nan_sorted.columns = ['Column', 'Number of NaNs']


# In[9]:


import seaborn as sns


# In[10]:


fig, ax = plt.subplots(figsize=(12, 25))
sns.barplot(x="Number of NaNs", y="Column", data=nan_sorted, color='Blue', ax=ax)
ax.set(xlabel="Number of NaNs", ylabel="", title="Total Number of NaNs in each column")
plt.show()


# There are several columns which have a very high proportion of missing values. It may be worth analysing these more closely.

# ### Monthly Effects on Target Variable

# In[11]:


train['transaction_month'] = pd.DatetimeIndex(train['transactiondate']).month
train.sort_values('transaction_month', axis=0, ascending=True, inplace=True)
print(train.head())

ax = sns.stripplot(x=train['transaction_month'], y=train['logerror'])


# For submission we are required to predict values for October, November and December. The differing distributions of the target variable over these months indicates that it may be useful to create an additional 'transaction_month' feature as shown above. Lets have a closer look at the distribution across only October, November and December.

# In[12]:


ax1 = sns.stripplot(x=train['transaction_month'][train['transaction_month'] > 9], y=train['logerror'])


# Proportion of Transactions in Each Month

# In[13]:


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

# In[14]:


#fill NaN values with -1 and encode object columns 
for x in prop.columns:
    prop[x] = prop[x].fillna(-1)

#many more parcelids in properties file, merge with training file
train = pd.merge(train, prop, on='parcelid', how='left')
print(train.head())
print('---------------------')
print(train.shape)


# In[15]:


for c in train[['transactiondate', 'hashottuborspa', 'propertycountylandusecode', 'propertyzoningdesc', 'fireplaceflag', 'taxdelinquencyflag']]:
    label = LabelEncoder()
    label.fit(list(train[c].values))
    train[c] = label.transform(list(train[c].values))

x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train['logerror']


# In[16]:


print(x_train.head())
print('------------')
print(y_train.head())


# In[17]:


rf = RandomForestRegressor(n_estimators=30, max_features=None)

rf.fit(x_train, y_train)

rf_importance = rf.feature_importances_


importance = pd.DataFrame()
importance['features'] = x_train.columns
importance['importance'] = rf_importance
print(importance.head())


# In[18]:


importance.sort_values('importance', axis=0, inplace=True, ascending=False)

print('------------')
print(importance.head())


# In[19]:


fig = plt.figure(figsize=(10, 4), dpi=100)
plt.bar(range(len(importance)), importance['importance'])
plt.title('Feature Importances')
plt.xlabel('Feature Name')
plt.ylabel('Importance')
plt.xticks(range(len(importance)), importance['features'], rotation=90)
plt.show()


# Here we see that the greatest importance in predicting the log-error comes from features involving taxes and geographical location of the property. Notably, the 'transaction_month' feature that was engineered earlier was the 12th most important feature. 

# In[20]:


test= test.rename(columns={'ParcelId': 'parcelid'}) 
#To make it easier for merging datasets on same column_id later

