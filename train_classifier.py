import sys
import os
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xgboost as xgb


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def etl_extract(sql_db_file):
	"""
	-Loads SQL database

	INPUT: >>>sql_db filename(str)
           >>>disaster_categories_file type(str)

    OUTPUT: >>>X (pandas Dataframe) features
            >>>y (pandas series) multi target labels 
	"""
	conn = create_engine('sqlite:///data/disaster_response.db')
	df = pd.read_sql_table('df', conn)
	# X = df['message'][:5]
	# y = df.drop(['id', 'message', 'original', 'genre_news', 'genre_social', 'genre'], axis = 1).iloc[:5,]
	X = df['message']
	y = df.drop(['id', 'message', 'original', 'genre_news', 'genre_social', 'genre'], axis = 1)
	column_names = y.columns.values
	return X, y, column_names

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'url_placeholder')
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stopwords_ = stopwords.words("english")
    words = [word for word in tokens if word not in stopwords_]
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens

def build_model_pipeline():
    pipeline = Pipeline([
        ('features', FeatureUnion([
                    ('text_pipeline', Pipeline([
                                ('vect', CountVectorizer(tokenizer=tokenize)),
                                ('tfidf', TfidfTransformer())
                            ])),
                    ('starting_verb', StartingVerbExtractor())
                        ])),
        ('clf', MultiOutputClassifier(xgb.XGBClassifier()))
                    ])

    param_grid = {'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
                  # 'features__text_pipeline__vect__max_df': (0.75, 1.0),
                  'clf__estimator__n_estimators': [100, 250, 500]
             }


    scorer = make_scorer(gridsCV_scorer)
    grid_model = GridSearchCV(pipeline, param_grid=param_grid, scoring=scorer, verbose=20)
    return grid_model

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            try:
            	_, first_tag = pos_tags[0]
            except IndexError:
            	pass
            else:
            	if first_tag in ['VB', 'VBP']:
            		return True
        return False
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
            

def eval_metrics(actual, predicted, column_names):
    """
    calculate evaluation metrics for model
    
    INPUT: 
    --actual : array of actual labels
    --predicted: array of predicted labels
    
    OUTPUT:
    --- Pandas Dataframe containing the accuracy, precision, recall 
        and f1 score for a given set of actual and predicted labels.
    """
    metrics = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    for i in range(len(column_names)):
        accuracy = accuracy_score(actual.iloc[:,i], predicted[:,i])
        precision = precision_score(actual.iloc[:,i], predicted[:,i], average='weighted')
        recall = recall_score(actual.iloc[:,i], predicted[:,i], average='weighted')
        f1 = f1_score(actual.iloc[:,i], predicted[:,i], average='weighted')
        
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        print(classification_report(actual.iloc[:,i], predicted[:,i]))
        
    df = pd.DataFrame(metrics, index=column_names)
    return df

def gridsCV_scorer(actual, predicted):
    """
    calculate median median f1 score for all output classifiers
    INPUT: 
    --actual : array of actual labels
    --predicted: array of predicted labels
    
    OUTPUT: Median Output Score
    """
    f1 = []
    for i in range(actual.shape[1]):
        f1_scr = f1_score(actual.iloc[:,i], predicted[:,i], average='weighted')
        f1.append(f1_scr)
        return np.median(f1)

def model_eval(model, X_test, y_test, label_names):
	"""
	calls eval_metircs functions that returns evaluation metrics 
	on test set
	INPUT: model
	       X_test (variables)
	       y_test (target labels)
	       label_names target label names
	"""
	predictions = model.predict(X_test)
	eval_metrics(y_test, predictions, label_names)

def save_model(model, file_path):
	"""
	serializing Model and save to disk
	"""
	with open(file_path, 'wb') as f:
		pickle.dump(model, f)

def main():
	if len(sys.argv) == 3:
		sql_db_file, model_file = sys.argv[1:]
		# print(f"Loading data...\nDATABASE: {sql_db_file} ")
		print("Loading data...\nDATABASE: {}".format(sql_db_file))
		X, y, column_names = etl_extract(sql_db_file)
		X_train, X_test, y_train, y_test = train_test_split(X, y)

		print("Build Model......")
		model = build_model_pipeline()

		print("Training Model")
		model.fit(X_train, y_train)

		print("Model Evaluation")
		model_eval(model, X_test, y_test, column_names)

		# print(f"Saving Model...\n  MODEL-->{model_file}")
		print("Saving Model...\n  MODEL-->{}".format(model_file))
		save_model(model, model_file)

	else:
		print("Please provide the filepaths of the messages and categories"\
              "datasets as the first and second argument respectively, as"\
              "well as the filepath of the database to save the cleaned data"\
              "as the third argument."\
              "\n\nExample: python train_classifier.py"\
              "data/DisasterResponse.db "\
              "DisasterResponse.pkl")

if __name__ == '__main__':
	main()