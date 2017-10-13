
# Zillow prize data analysis report

    This report was last updated on 2017-10-12 at 11:52:31


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

### Analysis

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
- xgboost
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

### Import Libraries and Data

Input data files are available in the "../input/" directory.

Any results I write to the current directory are saved as output.




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
      <th>Column</th>
      <th>percentNaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>storytypeid</td>
      <td>99.945599</td>
    </tr>
    <tr>
      <th>1</th>
      <td>basementsqft</td>
      <td>99.945465</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yardbuildingsqft26</td>
      <td>99.911330</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fireplaceflag</td>
      <td>99.827048</td>
    </tr>
    <tr>
      <th>4</th>
      <td>architecturalstyletypeid</td>
      <td>99.796966</td>
    </tr>
  </tbody>
</table>
</div>




![png](zillow_MWS_files/zillow_MWS_25_0.png)


There are several columns which have a very high proportion of missing values. I will remove features that have more than 80% missing values.

#### Feature Importance by Random Forest




    (90275, 3)



Feature Importance




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features=None, max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)



                       features  importance
    0         transaction_month    0.041078
    1     airconditioningtypeid    0.005809
    2  architecturalstyletypeid    0.000262
    3              basementsqft    0.000238
    4               bathroomcnt    0.007725


                          features  importance
    50  structuretaxvaluedollarcnt    0.084278
    24                    latitude    0.078829
    54                   taxamount    0.076839
    25                   longitude    0.072749
    26           lotsizesquarefeet    0.071840



![png](zillow_MWS_files/zillow_MWS_36_0.png)





    ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=30,
              max_features=0.3, max_leaf_nodes=None, min_impurity_decrease=0.0,
              min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=-1,
              oob_score=False, random_state=0, verbose=0, warm_start=False)



                       features  importance
    0         transaction_month    0.052949
    1     airconditioningtypeid    0.015291
    2  architecturalstyletypeid    0.000204
    3              basementsqft    0.000212
    4               bathroomcnt    0.014486


                          features  importance
    50  structuretaxvaluedollarcnt    0.060509
    54                   taxamount    0.059521
    53       landtaxvaluedollarcnt    0.056897
    51           taxvaluedollarcnt    0.056176
    26           lotsizesquarefeet    0.054112



![png](zillow_MWS_files/zillow_MWS_40_0.png)



![png](zillow_MWS_files/zillow_MWS_42_0.png)


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
