# Zillow Challenge

This document describes my Advanced Data Science I (140.711.01) final project.

For my project, I will compete for the [Zillow prize](https://www.zillow.com/promo/Zillow-prize/) and write up my results and experiences.

The data provided for the challenge are described at [Zillow prize site](https://www.zillow.com/promo/Zillow-prize/data).

## Primary objective

My **primary objective** is to describe my efforts to put together a half-decent entry in the Zillow challenge.

## Repository organization

The emphasis in this README file is the description of *process* of designing and planning this project, while the project files focus on the *process* of exploratory data analysis.

#### Project Files
|Name    | Type | Contains|
|:------------:|:---:|:---------:|
| 01_zillow_MWS.ipynb  | Main source   | Everything     |
| 01_zillow_MWS.py     | Python script | Text & Code    |
| 01_zillow_MWS.html   | Report        | Text & Figures |
| 01_zillow_MWS.md     | Report        | Text & Figures |
| 01_zillow_MWS.pdf    | Report        | Text & Figures |

The code for the report is saved as `01_zillow_MWS.ipynb` and `01_zillow_MWS.py` files. The report is saved as `01_zillow_MWS.md`, `01_zillow_MWS.html` and `01_zillow_MWS.pdf` files. Each time I save the notebook code and report files are automatically generated thanks to the save hook in the `jupyter_notebook_config.py` file. The config file also specifies that report files contain no code or input/output numbering.

## Reproducibility

I spent a great deal of time trying to figure how to make my data analysis report reproducible.
There are three strategies to run the code:
- Install Anaconda and create a conda environment using the the `env.yml` file. The `env.yml` file has a list of all packages and versions.
- Use the Kaggle Docker container.
- Run the code on Kaggle.

More info on these three options in the config folder!

## Environment
After signing up for the challenge, I decided to use only Python in the [Jupyter Notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html) environment. This is a personal preference, _de gustibus non est disputandum_. If I want to use, I can with the [Rpy2](https://rpy2.readthedocs.io) package, while the use of Python in an R Notebook in RStudio is extremely limited.

### First impressions
I was very happy to find out that Kaggle supports the Jupyter Notebook format. I can upload and download notebooks and work with them using the Kaggle Notebook interface, which is very similar to the Notebook interface with which I am deeply enamored (see previous section).


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
1. Create Kaggle account **Done**
2. Create a GitHub repo for the Advanced Data Science I (140.711.01) final project **Done**
3. Download the data files and put in the repo **Done**
4. Add the data files to `.gitignore` except for `zillow_data_dictionary.xlsx` (useful code book that explains the data) **Done**
5. Perform exploratory data analysis **Done**
5. Choose and implement imputation method **Done**
5. Split the training data into training and test sets **Done**
6. Try different algorithms using the Scikit-Learn **Done**
7. Measure algorithm performance 
7. Select the top algorithm(s)
8. Assess opportunities to improve performance of the top algorithm(s)
