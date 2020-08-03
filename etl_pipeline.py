import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def etl_extract(messages_file, disaster_categories_file):
	"""
	-Takes two CSV files, 
	-merges them as pandas Dataframe into one single Dataframe

	INPUT: >>>messages_file type(str)
           >>>disaster_categories_file type(str)

    OUTPUT: merged pandas dataframe
	"""
	disaster_categories = pd.read_csv(messages_file)
	disaster_messages = pd.read_csv(disaster_categories_file)
	df = pd.merge(disaster_messages, disaster_categories, on='id')
	return df

def etl_transform(df):
	"""
	-cleans(transforms) merged dataframe for ML, 
	
	INPUT: >>>df (pandas Dataframe) returned -> etl_extract()

    OUTPUT: cleaned pandas dataframe
	"""
	categories = df.categories.str.split(pat=';', expand=True)
	categories_names = categories.iloc[0,:].apply(lambda x: x[:-2])
	categories.columns = categories_names

	for column in categories:
		categories[column] = categories[column].str[-1].astype(int)

	df.drop('categories', 1, inplace=True)
	df = pd.concat([df, categories], axis=1)
	df.drop_duplicates(inplace=True)
	df=pd.concat([df, pd.get_dummies(df[['genre']], drop_first=True)], axis=1)
	# df.drop('genre', 1, inplace=True)

	return df

def etl_load(df, sql_database_filename):
	"""
	-saves df to SQL database, 
	
	INPUT: >>>df (pandas Dataframe) returned -> etl_transform()
	       >>> sql_database_filename (str): file path of sql database

    OUTPUT: None
	"""	
	conn = create_engine('sqlite:///data/disaster_response.db')
	df.to_sql('df', conn, index=False, if_exists='replace')
	pass

def main():
	if len(sys.argv) == 4:
		messages_file, disaster_categories, sql_database = sys.argv[1:]

		print("Loading data... \n Messages: {}\n CATEGORIES: {}".format(messages_file, disaster_categories))
		df = etl_extract(messages_file, disaster_categories)

		print("Cleaning Data......")
		df  = etl_transform(df)

		print("Saving data...\n DATABASE: {}".format(sql_database))
		etl_load(df, sql_database)
		print("Database Saved")

	else:
		print("Please provide the filepaths of the messages and categories"\
              "datasets as the first and second argument respectively, as "\
              "well as the filepath of the database to save the cleaned data "\
              "as the third argument."\
              "\n\nExample: python process_data.py"\
              "disaster_messages.csv disaster_categories.csv "\
              "DisasterResponse.db")

if __name__ == '__main__':
	main()