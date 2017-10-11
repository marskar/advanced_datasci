
# coding: utf-8

# # Zillow prize data analysis report

# In[1]:


from datetime import datetime
d = datetime.now().date()
t = datetime.now().strftime('%H:%M:%S')
print("This report was last updated on", d, "at", t)


# ## Introduction

# The [Zillow Prize](https://www.zillow.com/promo/Zillow-prize/) is a [Kaggle competition](https://www.kaggle.com/c/zillow-prize-1) that aims to inspire data scientists around the world to improve the accuracy of the Zillow "Zestimate" statistical and machine learning models. 
# 
# My goal is to compete for the Zillow prize and write up my results.

# ## Methods

# ### Data

# The data were obtained from [Kaggle website](https://www.kaggle.com/c/zillow-prize-1/data) and consist of the following files:
# - `properties_2016.csv.zip`
# - `properties_2017.csv.zip`
# - `sample_submission.csv`
# - `train_2016_v2.csv.zip`
# - `train_2017.csv.zip`
# - `zillow_data_dictionary.xlsx` 
# The `zillow_data_dictionary.xlsx` is a code book that explains the data.
# This data will be made available on [figshare](https://figshare.com/) to provide an additional source if the [Kaggle site data](https://www.kaggle.com/c/zillow-prize-1/data) become unavailable.

# ### Exploratory Data Analysis

# Data analysis was done in Jupyter Notebook (Pérez and Granger 2007)<cite data-cite="5251998/SH25XT8L"></cite> Integrated Development Environment using the Python language (Pérez, Granger, and Hunter 2011)<cite data-cite="5251998/FGTD82L2"></cite> and a number of software packages:
# 
# - NumPy (van der Walt, Colbert, and Varoquaux 2011)<cite data-cite="5251998/3SWILWGR"></cite>
# 
# - pandas (McKinney 2010)<cite data-cite="5251998/K3NZPGU9"></cite>
# 
# - scikit-learn (Pedregosa et al. 2011)<cite data-cite="5251998/SBYLEUVD"></cite>
# 
# 

# ### Visualization

# The following packages were used to visualize the data:
# 
# - Matplotlib (Hunter 2007)<cite data-cite="5251998/WP5LZ6AZ"></cite>
# 
# - Seaborn (Waskom et al. 2014)<cite data-cite="5251998/NSFX6VMN"></cite>
# 
# - r-ggplot2
# 
# - r-cowplot
# 
# The use of `R` code and packages in a `Python` environment is possible through the use of the `Rpy2` package.

# ### Prediction

# Machine learning prediction was done using the following packages:
# 
# - scikit-learn (Pedregosa et al. 2011)<cite data-cite="5251998/SBYLEUVD"></cite>
# 
# - r-caret 

# ### Reproducibility

# Reproducibility is extremely important in scientific research yet many examples of problematic studies exist in the literature (Couzin-Frankel 2010)<cite data-cite="5251998/UXR4ZTUS"></cite>.
# 
# The names and versions of each package used herein are listed in the accompanying `env.yml` file in the `config` folder.
# The computational environment used to analyze the data can be recreated using this `env.yml` file and the [`conda` package and environment manager](https://conda.io/docs/using/envs.html) available as part of the [Anaconda distribution of Python](https://www.anaconda.com/download/).
# 
# Additionally, details on how to setup a Docker image capable of running the analysis is included in the `README.md` file in the `config` folder.
# 
# The code in the form of a jupyter notebook (`01_zillow_MWS.ipynb`) or Python script (`01_zillow_MWS.py`), can also be run on the Kaggle website (this requires logging in with a username and password).
# 
# More information on the details of how this project was created and the computational environment was configured can be found in the accompanying `README.md` file.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python (a modified version of this docker image will be made available as part of my project to ensure reproducibility).
# For example, here's several helpful packages to load in 

# ## Results

# ### Import Libraries and Data for Exploratory Data Analysis

# In[3]:


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


# Input data files are available in the `../data/` directory.

# Any results I write to the current directory are saved as output.

# In[4]:


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


# In[7]:


### Let's import our data
data = pd.read_csv('../data/properties_2016.csv' , dtype = feature_dtypes)


# In[16]:


data.columns


# In[19]:


len(data.columns)


# In[8]:


continuous = ['basementsqft', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 
              'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
              'finishedsquarefeet50', 'finishedsquarefeet6', 'garagetotalsqft', 'latitude',
              'longitude', 'lotsizesquarefeet', 'poolsizesum',  'yardbuildingsqft17',
              'yardbuildingsqft26', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
              'landtaxvaluedollarcnt', 'taxamount']

discrete = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'fireplacecnt', 'fullbathcnt',
            'garagecarcnt', 'poolcnt', 'roomcnt', 'threequarterbathnbr', 'unitcnt',
            'numberofstories', 'assessmentyear', 'taxdelinquencyyear']


# In[17]:


len(continuous)


# In[18]:


len(discrete)


# ### Exploratory Data Analysis

# In[10]:


### Reading train file
errors = pd.read_csv('../data/train_2016_v2.csv', parse_dates=['transactiondate'])


# In[11]:


#### Merging tables
data_sold = data.merge(errors, how='inner', on='parcelid')


# In[12]:


### Creating 5 equal size logerror bins 
data_sold['logerror_bin'] = pd.qcut(data_sold['logerror'], 5, 
                                    labels=['Large Negative Error', 'Medium Negative Error',
                                            'Small Error', 'Medium Positive Error',
                                            'Large Positive Error'])
print(data_sold.logerror_bin.value_counts())


# In[13]:


### Continuous variable vs logerror plots
for col in continuous:     
    fig = plt.figure(figsize=(18,9));
    plt.xlabel('LogError Bin');
    plt.ylabel('Average {}'.format(col));
    sns.regplot(x='logerror', y=col, data=data_sold, color='Sienna', ax = plt.subplot(122));
    plt.suptitle('LogError vs {}'.format(col), fontsize=16)   


# ### Prediction

# In Progress

# ## Supplemental figures

# In Progress

# In[20]:


train_df = pd.read_csv("../data/train_2016_v2.csv", parse_dates=["transactiondate"])
train_df.shape


# In[21]:


train_y = train_df['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

from sklearn import ensemble


# Distribution of Target Variable:

# In[27]:


#load property features/description file
prop = pd.read_csv("../data/properties_2016.csv")
print(prop.shape)


# In[28]:


(train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts()


# In[29]:


prop_df = pd.read_csv("../data/properties_2016.csv")
prop_df.shape


# In[30]:


prop_df.head()


# In[31]:


train_df = pd.read_csv("../data/train_2016_v2.csv", parse_dates=["transactiondate"])
train_df.shape


# In[32]:


train_y = train_df['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

from sklearn import ensemble


# In[37]:


missing_df = prop_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# In[35]:


train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
train_df.head()


# In[38]:


pd.options.display.max_rows = 65

dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[39]:


dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[40]:


missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.999]


# In[41]:


# Let us just impute the missing values with mean values to compute correlation coefficients #
mean_values = train_df.mean(axis=0)
train_df_new = train_df.fillna(mean_values, inplace=True)

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
#autolabel(rects)
plt.show()


# In[42]:


corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']
for col in corr_zero_cols:
    print(col, len(train_df_new[col].unique()))


# In[43]:


corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]
corr_df_sel


# In[44]:


cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[45]:


col = "finishedsquarefeet12"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, size=10)
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Finished Square Feet 12', fontsize=12)
plt.title("Finished square feet 12 Vs Log error", fontsize=15)
plt.show()


# In[46]:


col = "calculatedfinishedsquarefeet"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.calculatedfinishedsquarefeet.values, y=train_df.logerror.values, size=10)
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Calculated finished square feet', fontsize=12)
plt.title("Calculated finished square feet Vs Log error", fontsize=15)
plt.show()


# In[47]:


plt.figure(figsize=(12,8))
sns.countplot(x="bathroomcnt", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bathroom', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bathroom count", fontsize=15)
plt.show()


# In[48]:


plt.figure(figsize=(12,8))
sns.countplot(x="bedroomcnt", data=train_df)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Bedroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bedroom count", fontsize=15)
plt.show()


# In[58]:


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


# In[59]:


# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, height=0.8, ax=ax)
plt.show()


# In[60]:


print("Training Size:" + str(train_df.shape))
print("Property Size:" + str(prop_df.shape))


# In[ ]:


### ... check for NaNs
nan = prop_df.isnull().sum()/prop_df.sum()
nan


# In[63]:


### Plotting NaN counts
nan_sorted = nan.sort_values(ascending=False).to_frame().reset_index()
nan_sorted.columns = ['Column', 'Number of NaNs']


# In[64]:


import seaborn as sns


# In[65]:


fig, ax = plt.subplots(figsize=(12, 25))
sns.barplot(x="Number of NaNs", y="Column", data=nan_sorted, color='Blue', ax=ax)
ax.set(xlabel="Number of NaNs", ylabel="", title="Total Number of NaNs in each column")
plt.show()


# Feature Missing Values and  Importance

# ## Conclusions

# In Progress

# ## Bibliography

# Couzin-Frankel, J. 2010. “Cancer Research. As Questions Grow, Duke Halts Trials, Launches Investigation.” Science 329 (5992): 614–15. 
# 
# Hunter, J. D. 2007. “Matplotlib: A 2D Graphics Environment.” Computing In Science & Engineering 9 (3): 90–95.
# 
# McKinney, W. 2010. “Data Structures for Statistical Computing in Python.” In Proceedings of the 9th Python in Science Conference, edited by S. J. van der Walt and K. J. Millman. Austin, Texas.
# 
# Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, et al. 2011. “Scikit-Learn: Machine Learning in Python.” Journal of Machine Learning Research 12 (Oct): 2825–30.
# 
# Pérez, F., and B. E. Granger. 2007. “IPython: A System for Interactive Scientific Computing.” Computing in Science & Engineering 9 (3): 21–29.
# 
# Pérez, F., B. E. Granger, and J. D. Hunter. 2011. “Python: An Ecosystem for Scientific Computing.” Computing in Science & Engineering 13 (2): 13–21.
# 
# Van der Walt, S., S. C. Colbert, and G. Varoquaux. 2011. “The NumPy Array: A Structure for Efficient Numerical Computation.” Computing in Science & Engineering 13 (2): 22–30.
# 
# Waskom, M, O Botvinnik, P Hobson, J Warmenhoven, JB Cole, Y Halchenko, J Vanderplas, et al. 2014. Seaborn: Statistical Data Visualization. Stanford, California.

# <div class="cite2c-biblio"></div>
