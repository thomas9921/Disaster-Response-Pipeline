import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    
    """
    Load data from database
    
    Args:
        database_filepath: path for sqlite database
    Return:
        X : features
        Y : labels
        category_names: labels for categories
    """
   
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_query('select * from cleanDF', engine)
    X = df['message'].values
    Y = df.iloc[:,5:]
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    
    """
    Tokenize the text strings
    
    Args:
        text: text string
    Return:
        clean_tokens: Array of tokenized strings
        
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
   
    clean_tokens =[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens 


def build_model():
    """
    Build pipeline for vectorizer, Tfidf, and Random forest
    
    Return: 
        cv: GridsearchCV
        
    """
    pipeline = Pipeline([
         ('vect', CountVectorizer(tokenizer=tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(RandomForestClassifier()))
         ])
    
    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 5],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
   
    """
    Evaluate the model against a test dataset
    Args:
        model: training model
        X_test: Test features
        Y_test: Testing labels
        category_names: labels for categories
    Return:
        Print: Accuracy scores
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=Y_test.keys()))

def save_model(model, model_filepath):
    """
    Save the model to a pickle
    Args:
        model: Training model
        model_filepath: Path to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
