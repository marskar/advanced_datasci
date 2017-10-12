
# coding: utf-8

# # Zillow prize data analysis report

# In[ ]:


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

# ### Analysis

# Data analysis was done in Jupyter Notebook (Pérez and Granger 2007)<cite data-cite="5251998/SH25XT8L"></cite> Integrated Development Environment using the Python language (Pérez, Granger, and Hunter 2011)<cite data-cite="5251998/FGTD82L2"></cite> and a number of software packages:
# 
# - NumPy (van der Walt, Colbert, and Varoquaux 2011)<cite data-cite="5251998/3SWILWGR"></cite>
# 
# - pandas (McKinney 2010)<cite data-cite="5251998/K3NZPGU9"></cite>
# 
# - scikit-learn (Pedregosa et al. 2011)<cite data-cite="5251998/SBYLEUVD"></cite>
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
# - xgboost
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

# ### Import Libraries and Data

# Input data files are available in the "../input/" directory.

# Any results I write to the current directory are saved as output.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import datetime as dt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
get_ipython().run_line_magic('matplotlib', 'inline')
### Seaborn style
sns.set_style("whitegrid")


# In[2]:


prop = pd.read_csv("../input/properties_2016.csv")
prop.shape


# In[3]:


### ... check for NaNs
nan = prop.isnull().sum()/len(prop)*100
nan


# In[4]:


### Plotting NaN counts
nan_sorted = nan.sort_values(ascending=False).to_frame().reset_index()
nan_sorted.columns = ['Column', 'Number of NaNs']


# In[5]:


fig, ax = plt.subplots(figsize=(12, 25))
sns.barplot(x="Number of NaNs", y="Column", data=nan_sorted, color='Blue', ax=ax)
ax.set(xlabel="Number of NaNs", ylabel="", title="Total Number of NaNs in each column")
plt.show()


# There are several columns which have a very high proportion of missing values. It may be worth analysing these more closely.

# #### Feature Importance by Random Forest

# In[6]:


train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
train.shape


# In[7]:


train['transaction_month'] = pd.DatetimeIndex(train['transactiondate']).month
train.sort_values('transaction_month', axis=0, ascending=True, inplace=True)


# For submission we are required to predict values for October, November and December. The differing distributions of the target variable over these months indicates that it may be useful to create an additional 'transaction_month' feature as shown above. Lets have a closer look at the distribution across only October, November and December.

# Proportion of Transactions in Each Month

# Feature Importance

# In[8]:


#fill NaN values with -1 and encode object columns 
for x in prop.columns:
    prop[x] = prop[x].fillna(-1)

#many more parcelids in properties file, merge with training file
train = pd.merge(train, prop, on='parcelid', how='left')


# In[9]:


for c in train[['transactiondate', 'hashottuborspa', 'propertycountylandusecode', 'propertyzoningdesc', 'fireplaceflag', 'taxdelinquencyflag']]:
    label = LabelEncoder()
    label.fit(list(train[c].values))
    train[c] = label.transform(list(train[c].values))

x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train['logerror']


# In[10]:


rf = RandomForestRegressor(n_estimators=30, max_features=None)

rf.fit(x_train, y_train)

rf_importance = rf.feature_importances_


importance = pd.DataFrame()
importance['features'] = x_train.columns
importance['importance'] = rf_importance
print(importance.head())


# In[11]:


importance.sort_values('importance', axis=0, inplace=True, ascending=False)

print('------------')
print(importance.head())


# In[12]:


fig = plt.figure(figsize=(10, 4), dpi=100)
plt.bar(range(len(importance)), importance['importance'])
plt.title('Feature Importances')
plt.xlabel('Feature Name')
plt.ylabel('Importance')
plt.xticks(range(len(importance)), importance['features'], rotation=90)
plt.show()


# In[13]:



xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)


# In[14]:


# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, height=0.8, ax=ax)
plt.show()


# In[16]:


train_y = train['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train = train.drop(['parcelid', 'logerror', 'transactiondate']+cat_cols, axis=1)
feat_names = train.columns.values


model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
model.fit(train, train_y)

## plot the importances ##
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# Here we see that the greatest importance in predicting the log-error comes from features involving taxes and geographical location of the property. Notably, the 'transaction_month' feature that was engineered earlier was the 12th most important feature. 

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
