# imports

from collections import Counter
import json, plotly
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from plotly.graph_objs import Bar
import re
import joblib
from sqlalchemy import create_engine

# initializing Flask app
app = Flask(__name__)

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

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


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'url_placeholder')

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stopwords_ = stopwords.words("english")
    words = [word for word in tokens if word not in stopwords_]
    words = [WordNetLemmatizer().lemmatize(word) for word in words]
        
    return words

conn = create_engine('sqlite:///../nano/data/disaster_response.db')
df = pd.read_sql_table('df', conn)

# load model
model = joblib.load("../nano/DisasterResponse.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    categories_prop = df.\
    drop(['id', 'message', 'original', 'genre_news', 'genre_social', 'genre'], axis = 1).sum()/len(df)

    categories_prop = categories_prop.sort_values(ascending = False)

    categories = list(categories_prop.index)

    # create visuals
    figures = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=categories_prop
                )
            ],

            'layout': {
                'title': 'Proportion of Messages by Category',
                'yaxis': {
                    'title': "Proportion",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -40,
                    'automargin':True
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly figures
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, data_set=df)

# web page that handles user query and displays model results
@app.route('/go')

def go():

    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(
        zip(df.drop(['id', 'message', 'original', 'genre_news', 'genre_social', 'genre'], axis = 1)\
            .columns, classification_labels))

    # This will render the go.html
    return render_template('go.html',
                            query=query,
                            classification_result=classification_results
                          )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()