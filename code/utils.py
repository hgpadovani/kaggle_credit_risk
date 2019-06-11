import numpy as np
import pandas as pd

def drop_missings_columns(df, treshold = 0.9):
	'''
	Function that calculates all columns that have missing data above a certain treshold
	Parameters:
	-----------
	Inputs:
	    df(Pandas Dataframe): Dataframe to compute the columns to drop
	    thresold(float): limiar to calculate
	Returns:
	    drop_columns (list): list with columns to drop
	    df_no_missings (pandas dataframe): Dataframe without dropped columns
	'''

	total_samples = df.shape[0]

	missings_per_column = df.isnull().sum()

	percentages_per_column = missings_per_column / total_samples
	drop_columns = list(percentages_per_column[percentages_per_column > treshold].index)

	return df.drop(drop_columns, axis=1), drop_columns

def fill_numerical_missings(df1, method='mean'):
	'''
	Function that fills missing data
	Parameters:
	-----------
	Inputs:
	    df1 (pandas dataframe): numerical dataframe to fill missings
	    method (string): string method
	Returns:
	    df (filled pandas dataframe): pandas dataframe with missing treatment
	'''

	df = df1.copy()

	if method == 'mean':
	    for col in df.columns:
	        df[col].fillna(value = df[col].mean(), inplace=True)
	elif method == 'median':
	    for col in df.columns:
	        df[col].fillna(value = df[col].median(), inplace=True)
	        
	return df
        
    

def fill_categorical_missings(df1, method='mode'):
	'''
	Function that fills missing data
	Parameters:
	-----------
	Inputs:
	    df1 (pandas dataframe): numerical dataframe to fill missings
	    method (string): string method
	Returns:
	    df (filled pandas dataframe): pandas dataframe with missing treatment
	'''

	df = df1.copy()

	for col in df.columns:
	    df[col].fillna(value = df[col].mode().values[0], inplace=True)

	return df
        
 
def get_catcodes(df):
	'''
	Function that create catcodes to all categorical columns
	Parameters:
	-----------
	Inputs:
	    df(pandas dataframe): categorical dataframe to apply the factorize function
	Returns:
	    df_dummed(pandas dataframe): datafrmar with categorical variables dummyfied
	'''
	mappers = {}
	for col in df.columns:
	    df[col], uniques = pd.factorize(df[col])
	    mappers[col] = dict(zip(uniques, df[col].unique()))
	    
	return df, mappers


def objective_lgbm(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    
    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    
    run_time = timer() - start
    
     # Extract the best score
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}


def objective(space, n_folds = N_FOLDS):
    """Objective function for Generic Model Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
        
    # KFold cross-validation
    kf = StratifiedKFold(n_splits = N_FOLDS)
    
    # Getting the model
    model = space['model'](**space['params'])

    # Starting
    start = timer()
    
    # Perform n_folds cross validation
    roc_aucs = []
    for train_index, val_index in kf.split(X_train, y_train):
        X_train_, y_train_ = X_train[train_index, :], y_train[train_index]
        X_val_, y_val_ = X_train[val_index, :], y_train[val_index]

        # Change the model here
        model.fit(X_train_, y_train_)
        roc_aucs.append(roc_auc_score(y_val_, model.predict_proba(X_val_)[:,1]))
    
    run_time = timer() - start
    
     # Extract the best score
    best_score = np.mean(roc_aucs)
    
    # Loss must be minimized
    loss = 1 - best_score

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}

def optimize(objective, space, MAX_EVALS = 120, trials, output_file = 'BayesOpt.csv', random_state = 42):
    
    # Output file
    out_file = output_file
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)

    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration', 'train_time'])
    of_connection.close()
    
    # Global variable
    global  ITERATION

    ITERATION = 0

    # Run optimization
    best = fmin(fn = objective, space = space, algo = tpe.suggest, 
                max_evals = MAX_EVALS, trials = trials, rstate = random_state)
    
    return space_eval(space, best)
    