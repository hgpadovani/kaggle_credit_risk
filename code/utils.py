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