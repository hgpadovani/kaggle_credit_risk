# Kaggle competition: Home Credit Default Risk.

Link to the competition [here](https://www.kaggle.com/c/home-credit-default-risk).

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

This is the main goal of this competition: to correct predict weather a client is able to pay a loan based on previous loans.

# Evaluation

The submissions are going to be evaluated on area under the ROC curve (or AUC) between predicted probability and the observed target. 


# Data

 - application_{train|test}.csv:

	- This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
	- Static data for all applications. One row represents one loan in our data sample.

- bureau.csv

	- All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
	- For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

- bureau_balance.csv

	- Monthly balances of previous credits in Credit Bureau.
	- This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

- POS_CASH_balance.csv

	- Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
	- This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

- credit_card_balance.csv

	- Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
	- This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

- previous_application.csv

	- All previous applications for Home Credit loans of clients who have loans in our sample.
	- There is one row for each previous application related to loans in our data sample.

- installments_payments.csv

	- Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
	- There is a) one row for every payment that was made plus b) one row each for missed payment.
	- One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

- HomeCredit_columns_description.csv

	- This file contains descriptions for the columns in the various data files.

# Repo structure

I've organized this repo using the following structure:

- code
	- utils.py -> file containing functions and helpers
- data
	- raw -> folder containing raw data
	- treated -> folder containing clean/treated/engineered data
- notebooks
	- 1 - Fail Fast - First Contact -> notebook where I've started to play with the data, write a few functions to help me clean the data (filling missing values and dealing with categorical variables), and I've tested a few Machine Learning algorithms just to get a baseline submission.
	- 2 - EDA - Notebook where I do some Exploratory Data Analysis. I've plotted a lot of variable distributions, calculated correlations, and did a lot of explorations on the data. This is a very important step in order to gain familiarity with the data and to get some insights on what can be used to build a good model.
	- 3 - File_Exploration -> This notebooks aims to explore all the files provided by Kaggle. I've merged them all and did some cleaning.
	- 4 - Manual_Feature_Engineering -> Got some feature engineering done using some expansions. A few other domain knowledge features were constructed.
	- 5 - Automated_Feature_Engineering -> Automated feature engineering using Deep Feature Syntesis with featuretools lib.
	- 6 - Machine_Learning -> Notebook where I focused in applying several algorithms to the data. I've kept using tree based algorithms and its ensembles, and also created a few submission files with LightGBM, GradientBoostingClassifier, and (tried with) XGBoost.
	- 7 - Hyperparameter_Tunning -> Bayesian Optimisation pipeline. I've used the TPE algorithm to tune LightGBM's parameters.
	- 8 - Neural Nets -> This notebook brings neural network implementation. 
- spaces
	- search_spaces.py -> file where I've kept search spaces for a few algorithms.
- submissions
	- folder where I've stored my submissions.
- trials
	- folder where I've stored trials of bayesian optimisation.

