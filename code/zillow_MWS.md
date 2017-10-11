
# Zillow prize data analysis report

    This report was last updated on 2017-10-11 at 12:04:56


## Introduction

The [Zillow Prize](https://www.zillow.com/promo/Zillow-prize/) is a [Kaggle competition](https://www.kaggle.com/c/zillow-prize-1) that aims to inspire data scientists around the world to improve the accuracy of the Zillow "Zestimate" statistical and machine learning models. 

My goal is to compete for the Zillow prize and write up my results.

## Methods

### Data

The data were obtained from [Kaggle website](https://www.kaggle.com/c/zillow-prize-1/data) and consist of the following files:
- `properties_2016.csv.zip`
- `properties_2017.csv.zip`
- `sample_submission.csv`
- `train_2016_v2.csv.zip`
- `train_2017.csv.zip`
- `zillow_data_dictionary.xlsx` 
The `zillow_data_dictionary.xlsx` is a code book that explains the data.
This data will be made available on [figshare](https://figshare.com/) to provide an additional source if the [Kaggle site data](https://www.kaggle.com/c/zillow-prize-1/data) become unavailable.

### Exploratory Data Analysis

Data analysis was done in Jupyter Notebook (Pérez and Granger 2007)<cite data-cite="5251998/SH25XT8L"></cite> Integrated Development Environment using the Python language (Pérez, Granger, and Hunter 2011)<cite data-cite="5251998/FGTD82L2"></cite> and a number of software packages:

- NumPy (van der Walt, Colbert, and Varoquaux 2011)<cite data-cite="5251998/3SWILWGR"></cite>

- pandas (McKinney 2010)<cite data-cite="5251998/K3NZPGU9"></cite>

- scikit-learn (Pedregosa et al. 2011)<cite data-cite="5251998/SBYLEUVD"></cite>



### Visualization

The following packages were used to visualize the data:

- Matplotlib (Hunter 2007)<cite data-cite="5251998/WP5LZ6AZ"></cite>

- Seaborn (Waskom et al. 2014)<cite data-cite="5251998/NSFX6VMN"></cite>

- r-ggplot2

- r-cowplot

The use of `R` code and packages in a `Python` environment is possible through the use of the `Rpy2` package.

### Prediction

Machine learning prediction was done using the following packages:

- scikit-learn (Pedregosa et al. 2011)<cite data-cite="5251998/SBYLEUVD"></cite>

- r-caret 

### Reproducibility

Reproducibility is extremely important in scientific research yet many examples of problematic studies exist in the literature (Couzin-Frankel 2010)<cite data-cite="5251998/UXR4ZTUS"></cite>.

The names and versions of each package used herein are listed in the accompanying `env.yml` file in the `config` folder.
The computational environment used to analyze the data can be recreated using this `env.yml` file and the [`conda` package and environment manager](https://conda.io/docs/using/envs.html) available as part of the [Anaconda distribution of Python](https://www.anaconda.com/download/).

Additionally, details on how to setup a Docker image capable of running the analysis is included in the `README.md` file in the `config` folder.

The code in the form of a jupyter notebook (`01_zillow_MWS.ipynb`) or Python script (`01_zillow_MWS.py`), can also be run on the Kaggle website (this requires logging in with a username and password).

More information on the details of how this project was created and the computational environment was configured can be found in the accompanying `README.md` file.

This Python 3 environment comes with many helpful analytics libraries installed
It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python (a modified version of this docker image will be made available as part of my project to ensure reproducibility).
For example, here's several helpful packages to load in 

## Results

### Import Libraries and Data for Exploratory Data Analysis

Input data files are available in the `../data/` directory.

Any results I write to the current directory are saved as output.




    Index(['parcelid', 'airconditioningtypeid', 'architecturalstyletypeid',
           'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid',
           'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid',
           'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
           'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
           'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
           'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'hashottuborspa',
           'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
           'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
           'propertycountylandusecode', 'propertylandusetypeid',
           'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',
           'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
           'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
           'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
           'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt',
           'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
           'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear',
           'censustractandblock'],
          dtype='object')






    58






    20






    13



### Exploratory Data Analysis




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parcelid</th>
      <th>logerror</th>
      <th>transactiondate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11016594</td>
      <td>0.0276</td>
      <td>2016-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14366692</td>
      <td>-0.1684</td>
      <td>2016-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12098116</td>
      <td>-0.0040</td>
      <td>2016-01-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12643413</td>
      <td>0.0218</td>
      <td>2016-01-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14432541</td>
      <td>-0.0050</td>
      <td>2016-01-02</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parcelid</th>
      <th>airconditioningtypeid</th>
      <th>architecturalstyletypeid</th>
      <th>basementsqft</th>
      <th>bathroomcnt</th>
      <th>bedroomcnt</th>
      <th>buildingclasstypeid</th>
      <th>buildingqualitytypeid</th>
      <th>calculatedbathnbr</th>
      <th>decktypeid</th>
      <th>...</th>
      <th>structuretaxvaluedollarcnt</th>
      <th>taxvaluedollarcnt</th>
      <th>assessmentyear</th>
      <th>landtaxvaluedollarcnt</th>
      <th>taxamount</th>
      <th>taxdelinquencyflag</th>
      <th>taxdelinquencyyear</th>
      <th>censustractandblock</th>
      <th>logerror</th>
      <th>transactiondate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17073783</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>NaN</td>
      <td>...</td>
      <td>115087.0</td>
      <td>191811.0</td>
      <td>2015.0</td>
      <td>76724.0</td>
      <td>2015.06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61110022003007</td>
      <td>0.0953</td>
      <td>2016-01-27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17088994</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>143809.0</td>
      <td>239679.0</td>
      <td>2015.0</td>
      <td>95870.0</td>
      <td>2581.30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61110015031002</td>
      <td>0.0198</td>
      <td>2016-03-30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17100444</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>33619.0</td>
      <td>47853.0</td>
      <td>2015.0</td>
      <td>14234.0</td>
      <td>591.64</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61110007011007</td>
      <td>0.0060</td>
      <td>2016-05-27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17102429</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>NaN</td>
      <td>...</td>
      <td>45609.0</td>
      <td>62914.0</td>
      <td>2015.0</td>
      <td>17305.0</td>
      <td>682.78</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61110008002013</td>
      <td>-0.0566</td>
      <td>2016-06-07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17109604</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>NaN</td>
      <td>...</td>
      <td>277000.0</td>
      <td>554000.0</td>
      <td>2015.0</td>
      <td>277000.0</td>
      <td>5886.92</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61110014021007</td>
      <td>0.0573</td>
      <td>2016-08-08</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>



    Large Negative Error     18442
    Small Error              18432
    Medium Negative Error    17973
    Large Positive Error     17947
    Medium Positive Error    17481
    Name: logerror_bin, dtype: int64



![png](zillow_MWS_files/zillow_MWS_32_0.png)



![png](zillow_MWS_files/zillow_MWS_32_1.png)



![png](zillow_MWS_files/zillow_MWS_32_2.png)



![png](zillow_MWS_files/zillow_MWS_32_3.png)



![png](zillow_MWS_files/zillow_MWS_32_4.png)



![png](zillow_MWS_files/zillow_MWS_32_5.png)



![png](zillow_MWS_files/zillow_MWS_32_6.png)



![png](zillow_MWS_files/zillow_MWS_32_7.png)



![png](zillow_MWS_files/zillow_MWS_32_8.png)



![png](zillow_MWS_files/zillow_MWS_32_9.png)



![png](zillow_MWS_files/zillow_MWS_32_10.png)



![png](zillow_MWS_files/zillow_MWS_32_11.png)



![png](zillow_MWS_files/zillow_MWS_32_12.png)



![png](zillow_MWS_files/zillow_MWS_32_13.png)



![png](zillow_MWS_files/zillow_MWS_32_14.png)



![png](zillow_MWS_files/zillow_MWS_32_15.png)



![png](zillow_MWS_files/zillow_MWS_32_16.png)



![png](zillow_MWS_files/zillow_MWS_32_17.png)



![png](zillow_MWS_files/zillow_MWS_32_18.png)



![png](zillow_MWS_files/zillow_MWS_32_19.png)


### Prediction

In Progress

## Supplemental figures

In Progress




    (90275, 3)



Distribution of Target Variable:

    /Users/marskar/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2728: DtypeWarning: Columns (22,32,34,49,55) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


       parcelid  airconditioningtypeid  architecturalstyletypeid  basementsqft  \
    0  10754147                    NaN                       NaN           NaN   
    1  10759547                    NaN                       NaN           NaN   
    2  10843547                    NaN                       NaN           NaN   
    3  10859147                    NaN                       NaN           NaN   
    4  10879947                    NaN                       NaN           NaN   
    
       bathroomcnt  bedroomcnt  buildingclasstypeid  buildingqualitytypeid  \
    0          0.0         0.0                  NaN                    NaN   
    1          0.0         0.0                  NaN                    NaN   
    2          0.0         0.0                  NaN                    NaN   
    3          0.0         0.0                  3.0                    7.0   
    4          0.0         0.0                  4.0                    NaN   
    
       calculatedbathnbr  decktypeid         ...           numberofstories  \
    0                NaN         NaN         ...                       NaN   
    1                NaN         NaN         ...                       NaN   
    2                NaN         NaN         ...                       NaN   
    3                NaN         NaN         ...                       1.0   
    4                NaN         NaN         ...                       NaN   
    
       fireplaceflag  structuretaxvaluedollarcnt  taxvaluedollarcnt  \
    0            NaN                         NaN                9.0   
    1            NaN                         NaN            27516.0   
    2            NaN                    650756.0          1413387.0   
    3            NaN                    571346.0          1156834.0   
    4            NaN                    193796.0           433491.0   
    
       assessmentyear  landtaxvaluedollarcnt  taxamount  taxdelinquencyflag  \
    0          2015.0                    9.0        NaN                 NaN   
    1          2015.0                27516.0        NaN                 NaN   
    2          2015.0               762631.0   20800.37                 NaN   
    3          2015.0               585488.0   14557.57                 NaN   
    4          2015.0               239695.0    5725.17                 NaN   
    
       taxdelinquencyyear  censustractandblock  
    0                 NaN                  NaN  
    1                 NaN                  NaN  
    2                 NaN                  NaN  
    3                 NaN                  NaN  
    4                 NaN                  NaN  
    
    [5 rows x 58 columns]
    ---------------------
    (2985217, 58)





    1    90026
    2      123
    3        1
    Name: parcelid, dtype: int64



    /Users/marskar/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2728: DtypeWarning: Columns (22,32,34,49,55) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





    (2985217, 58)






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parcelid</th>
      <th>airconditioningtypeid</th>
      <th>architecturalstyletypeid</th>
      <th>basementsqft</th>
      <th>bathroomcnt</th>
      <th>bedroomcnt</th>
      <th>buildingclasstypeid</th>
      <th>buildingqualitytypeid</th>
      <th>calculatedbathnbr</th>
      <th>decktypeid</th>
      <th>...</th>
      <th>numberofstories</th>
      <th>fireplaceflag</th>
      <th>structuretaxvaluedollarcnt</th>
      <th>taxvaluedollarcnt</th>
      <th>assessmentyear</th>
      <th>landtaxvaluedollarcnt</th>
      <th>taxamount</th>
      <th>taxdelinquencyflag</th>
      <th>taxdelinquencyyear</th>
      <th>censustractandblock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10754147</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>2015.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10759547</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27516.0</td>
      <td>2015.0</td>
      <td>27516.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10843547</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>650756.0</td>
      <td>1413387.0</td>
      <td>2015.0</td>
      <td>762631.0</td>
      <td>20800.37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10859147</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>571346.0</td>
      <td>1156834.0</td>
      <td>2015.0</td>
      <td>585488.0</td>
      <td>14557.57</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10879947</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>193796.0</td>
      <td>433491.0</td>
      <td>2015.0</td>
      <td>239695.0</td>
      <td>5725.17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>






    (90275, 3)




![png](zillow_MWS_files/zillow_MWS_46_0.png)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parcelid</th>
      <th>logerror</th>
      <th>transactiondate</th>
      <th>airconditioningtypeid</th>
      <th>architecturalstyletypeid</th>
      <th>basementsqft</th>
      <th>bathroomcnt</th>
      <th>bedroomcnt</th>
      <th>buildingclasstypeid</th>
      <th>buildingqualitytypeid</th>
      <th>...</th>
      <th>numberofstories</th>
      <th>fireplaceflag</th>
      <th>structuretaxvaluedollarcnt</th>
      <th>taxvaluedollarcnt</th>
      <th>assessmentyear</th>
      <th>landtaxvaluedollarcnt</th>
      <th>taxamount</th>
      <th>taxdelinquencyflag</th>
      <th>taxdelinquencyyear</th>
      <th>censustractandblock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11016594</td>
      <td>0.0276</td>
      <td>2016-01-01</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>122754.0</td>
      <td>360170.0</td>
      <td>2015.0</td>
      <td>237416.0</td>
      <td>6735.88</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.037107e+13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14366692</td>
      <td>-0.1684</td>
      <td>2016-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>346458.0</td>
      <td>585529.0</td>
      <td>2015.0</td>
      <td>239071.0</td>
      <td>10153.02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12098116</td>
      <td>-0.0040</td>
      <td>2016-01-01</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61994.0</td>
      <td>119906.0</td>
      <td>2015.0</td>
      <td>57912.0</td>
      <td>11484.48</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.037464e+13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12643413</td>
      <td>0.0218</td>
      <td>2016-01-02</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>171518.0</td>
      <td>244880.0</td>
      <td>2015.0</td>
      <td>73362.0</td>
      <td>3048.74</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.037296e+13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14432541</td>
      <td>-0.0050</td>
      <td>2016-01-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>169574.0</td>
      <td>434551.0</td>
      <td>2015.0</td>
      <td>264977.0</td>
      <td>5488.96</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.059042e+13</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Type</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>int64</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>float64</td>
      <td>53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>datetime64[ns]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>object</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Type</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>int64</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>float64</td>
      <td>53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>datetime64[ns]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>object</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



    /Users/marskar/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:4: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      after removing the cwd from sys.path.





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column_name</th>
      <th>missing_count</th>
      <th>missing_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>basementsqft</td>
      <td>90232</td>
      <td>0.999524</td>
    </tr>
    <tr>
      <th>8</th>
      <td>buildingclasstypeid</td>
      <td>90259</td>
      <td>0.999823</td>
    </tr>
    <tr>
      <th>15</th>
      <td>finishedsquarefeet13</td>
      <td>90242</td>
      <td>0.999634</td>
    </tr>
    <tr>
      <th>43</th>
      <td>storytypeid</td>
      <td>90232</td>
      <td>0.999524</td>
    </tr>
  </tbody>
</table>
</div>



    /Users/marskar/anaconda3/lib/python3.6/site-packages/numpy/lib/function_base.py:3162: RuntimeWarning: invalid value encountered in true_divide
      c /= stddev[:, None]
    /Users/marskar/anaconda3/lib/python3.6/site-packages/numpy/lib/function_base.py:3163: RuntimeWarning: invalid value encountered in true_divide
      c /= stddev[None, :]



![png](zillow_MWS_files/zillow_MWS_51_1.png)


    assessmentyear 1
    storytypeid 1
    pooltypeid2 1
    pooltypeid7 1
    pooltypeid10 1
    poolcnt 1
    decktypeid 1
    buildingclasstypeid 1


    /Users/marskar/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_labels</th>
      <th>corr_values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>heatingorsystemtypeid</td>
      <td>-0.019648</td>
    </tr>
    <tr>
      <th>45</th>
      <td>structuretaxvaluedollarcnt</td>
      <td>0.021935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bedroomcnt</td>
      <td>0.025467</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bathroomcnt</td>
      <td>0.027889</td>
    </tr>
    <tr>
      <th>18</th>
      <td>fullbathcnt</td>
      <td>0.028169</td>
    </tr>
    <tr>
      <th>7</th>
      <td>calculatedbathnbr</td>
      <td>0.028758</td>
    </tr>
    <tr>
      <th>10</th>
      <td>calculatedfinishedsquarefeet</td>
      <td>0.038341</td>
    </tr>
    <tr>
      <th>11</th>
      <td>finishedsquarefeet12</td>
      <td>0.039248</td>
    </tr>
  </tbody>
</table>
</div>




![png](zillow_MWS_files/zillow_MWS_54_0.png)


    /Users/marskar/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:4: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      after removing the cwd from sys.path.
    /Users/marskar/anaconda3/lib/python3.6/site-packages/pandas/core/indexing.py:179: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._setitem_with_indexer(indexer, value)



    <matplotlib.figure.Figure at 0x116e05630>



![png](zillow_MWS_files/zillow_MWS_55_2.png)


    /Users/marskar/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:4: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      after removing the cwd from sys.path.
    /Users/marskar/anaconda3/lib/python3.6/site-packages/pandas/core/indexing.py:179: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._setitem_with_indexer(indexer, value)



    <matplotlib.figure.Figure at 0x113b7c080>



![png](zillow_MWS_files/zillow_MWS_56_2.png)



![png](zillow_MWS_files/zillow_MWS_57_0.png)



![png](zillow_MWS_files/zillow_MWS_58_0.png)



![png](zillow_MWS_files/zillow_MWS_60_0.png)


    Training Size:(90275, 52)
    Property Size:(2985217, 58)



![png](zillow_MWS_files/zillow_MWS_65_0.png)


Feature Missing Values and  Importance

## Conclusions

In Progress

## Bibliography

Couzin-Frankel, J. 2010. “Cancer Research. As Questions Grow, Duke Halts Trials, Launches Investigation.” Science 329 (5992): 614–15. 

Hunter, J. D. 2007. “Matplotlib: A 2D Graphics Environment.” Computing In Science & Engineering 9 (3): 90–95.

McKinney, W. 2010. “Data Structures for Statistical Computing in Python.” In Proceedings of the 9th Python in Science Conference, edited by S. J. van der Walt and K. J. Millman. Austin, Texas.

Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, et al. 2011. “Scikit-Learn: Machine Learning in Python.” Journal of Machine Learning Research 12 (Oct): 2825–30.

Pérez, F., and B. E. Granger. 2007. “IPython: A System for Interactive Scientific Computing.” Computing in Science & Engineering 9 (3): 21–29.

Pérez, F., B. E. Granger, and J. D. Hunter. 2011. “Python: An Ecosystem for Scientific Computing.” Computing in Science & Engineering 13 (2): 13–21.

Van der Walt, S., S. C. Colbert, and G. Varoquaux. 2011. “The NumPy Array: A Structure for Efficient Numerical Computation.” Computing in Science & Engineering 13 (2): 22–30.

Waskom, M, O Botvinnik, P Hobson, J Warmenhoven, JB Cole, Y Halchenko, J Vanderplas, et al. 2014. Seaborn: Statistical Data Visualization. Stanford, California.

<div class="cite2c-biblio"></div>
