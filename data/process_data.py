# import libraries
import numpy as np
import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load both the categories.csv and messages.csv from input filepaths
    
    Args:
        messages_filepath: Filepath to the messages dataset
        categories_filepath: Filepath to the categories dataset
    Returns:
        (DataFrame) df: Merged dataframe with messages and categories          
    """
    # Read in datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #Merge the datasets
    df = pd.merge(messages, categories)
    
    return df

def clean_data(df):
    """
    Clean the merged dataset
    
    Args:
        df: dataframe from load_data
    Return:
        df: Cleaned dataframe
    """
    #Split categories into separate columns
    categories = df['categories'].str.split(';',expand=True)
    row = categories[:1]
    category_colnames = row.apply(lambda x: x.str.split('-')[0][0], axis=0)
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    #Replace categories column in df with new category columns
    df.drop('categories',axis=1,inplace=True)
    
    #Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df,categories],axis=1)
    
    #drop duplicates
    df = df.drop_duplicates(keep='first')
    
    return df
    
def save_data(df, database_filename):
    """
    Save dataset to sqlite database
    
    Args:
        df:Cleaned dataframe
        database_filename: Name of the database file
        
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('cleanDF', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()