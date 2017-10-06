# Zillow Challenge

This  document describes my plan for the Advanced Data Science I (140.711.01) final project.

For my project, I will compete for the [Zillow prize](https://www.zillow.com/promo/Zillow-prize/) and write up my results.

The data provided for the challenge are described at [Zillow prize site](https://www.zillow.com/promo/Zillow-prize/data).

## Primary objective
My **primary objective** is to describe my efforts to put together a half-decent entry in the Zillow challenge. The emphasis here will be on the *process* and exploratory data analysis.

## Reproducibility

I spent a great deal of time trying to figure how to 


## Secondary objectives
After signing up for the challenge, I decided to try things out in **BOTH** R and Python using the [Rpy2](https://rpy2.readthedocs.io) package in the [Jupyter Notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html) environment. This is a personal preference, but it prompted me to outline several secondary objectives:

1. Compare Machine Learning in Python and R
2. Compare Pandas and dplyr packages
3. Compare Scikit-Learn and Caret packages

<!--The Secondary Objectives are incloded in Table 1 below.

### Table 1.
|Subject    <td colspan="2"> <center> Comparison Area </center> </td> |
|:------------:|:---:|:---------:|
|    **Language**    |  Python   | R         |
|    **Wrangling**    | Pandas | dplyr     |
| **Machine Learning** | Scikit-Learn | Caret     |
|     **IDE**         | JupyterLab    | RStudio   |
|     **File type**    |  ipynb  | [R Markdown](http://rmarkdown.rstudio.com) | -->

### First impressions
I was very happy to find out that Kaggle supports the Jupyter Notebook format. I can upload and download notebooks and work with them using the Kaggle Notebook interface, which is very similar to the Notebook interface with which I am deeply enamored. The kaggle/python environment is defined by this [docker](https://www.docker.com/) image: https://github.com/kaggle/docker-python.


Zillow Prize Metric
-----------------------------
The challenge entails training a machine learning algorithm to predict the log error between Zillow's proprietary Zestimate prediction of home values and the actual home values.

The metric by which submissions are evaluated is the Mean Absolute Error between the predicted log error and the actual log error. The log error is defined as

$logerror=log(Zestimate) - log(SalePrice)$

Zillow Prize Timeline
----------------------------
- Release of 2017 Training data: 10/2/2017
- Round 1 Submission Deadline: 10/16/2017 11:59 PM PT

First steps
--------------
1. Create Kaggle and GitHub accounts **Done**
2. Create a GitHub repo for the Advanced Data Science I (140.711.01) final project.
3. Download the data files and put in the repo
4. Add the data files to .gitignore except for zillow_data_dictionary.xlsx
5. Perform exploratory data analysis
5. Split the training data into training and test sets
6. Try different algorithms using the Scikit-Learn and caret packages
7. Measure algorithm performance
7. Select the top algorithm(s)
8. Assess opportunities to improve performance of the top algorithm(s)


File management
------------
Last week I made a Kaggle account and downloaded the data files. I added the data files to .gitignore except for zillow_data_dictionary.xlsx, which is a useful code book that explains the data.

Next steps
---------
My next task is to explore the data, figure out what to do about missing data, and split the data into training and test sets.

Specifically, I plan to split the data into two groups
randomly, where 2/3 of the data will be used for training and 1/3 will be used for testing. I am not sure if I want to set up a cross-validation experiment. Perhaps a 10-fold or 5 * 2 cross-validation.
