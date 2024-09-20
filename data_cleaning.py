'''
Module Description: 
This module allows you to clean and filter the yelp dataset on reviews to only include bad reviews of restaurants.
It takes for input the pkl data file produced from the load_df.py module.
The dataframe includes a 'tokens' column with a lemmatized version of the text, including the following modifications to text:
    - Converted to lower case
    - Removed non-alphabetic characters
    - Removed stop words
The dataframe is saved as yelp_data_cleaned.pkl in the data directory.

Author: Ashley ACKER

To load the pickle file to pandas df:   
    import pickle
    with open("data/yelp_data_cleaned.pkl", "rb") as f:
        df = pickle.load(f)    

'''

# Downloads & Imports 
import nltk
# nltk.download('omw-1.4')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

import os
import re
# import pandas as pd
import numpy as np
import pickle 

from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Constants
DATA_DIR = "data"
RAW_DATA_FILE_PATH = os.path.join(DATA_DIR,  'yelp_reviews_raw_df.pkl') 
CLEAN_DATA_FILE_PATH = os.path.join(DATA_DIR, 'yelp_data_cleaned.pkl') 


def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()


def clean_df(data_file_path):
    ''' 
    Clean reviews dataframe and including only resturants with bad reviews.
    
    Returns: clean df
    '''
    # load df
    # df = pd.read_csv(data_file_path)
    with open(data_file_path, "rb") as f:
        df = pickle.load(f) 
    
    # removing the few rows with NA values (57 found in categories, which is needed for selection)
    df = df.dropna(axis=0)
    
    # removing rows where the review is less than 10 words
    
    # removing rows where the review is less than 10 words
    df = df[ df['text'].apply(lambda x: len(x)>10) ]
    
    # creating dummy feature for reviews on restaurants to be used to filter data 
    df['Resto_dummy'] = df['categories'].apply(lambda x: 1 if 'Restaurant' in x else 0)
    
    # selecting only restaurants with bad reviews (stars = 1 or 2) that are in english
    df = df[ (df['stars']<=2) & df['Resto_dummy'] & (df['language']=='en') ]
    
    # create dummy for greater than 500 words to analyse value of longer reviews
    df['Plus_de_500_mots'] = df['text'].apply(lambda x: 1 if len(x.split())>500 else 0)
    
    # creating column with only alphabetic lower case characters using text
    df['cleaned_text'] = df['text'].apply(clean_text)

    return df


def get_wordnet_pos(pos_tags):
    '''
    adds pos tags to text, which is needed for lemmatize_text function
    '''
    output = np.asarray(pos_tags)
    for i in range(len(pos_tags)):
        if pos_tags[i][1].startswith('J'):
            output[i][1] = wordnet.ADJ
        elif pos_tags[i][1].startswith('V'):
            output[i][1] = wordnet.VERB
        elif pos_tags[i][1].startswith('R'):
            output[i][1] = wordnet.ADV
        else:
            output[i][1] = wordnet.NOUN
    return output


def lemmatize_text(quote, lem_text=True, remove_stop_words=True, tokenize=True):
    ''' 
    Takes a body of text as quote. provides option to lemmatize the tokens and remove stop words.
    Puts text in lower case and only keeps alphabetic characters.
    
    Returns: tokenized version of text (list of cleaned words)
    
    Dependency: requires get_wordnet_pos function 
    '''
    stop_words = stopwords.words("english")
    
    tokens = quote.lower()
    tokens = [t for t in tokens if t.isalpha()]
    
    if tokenize:
        tokens = word_tokenize(quote)
    if remove_stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    if lem_text:
        tags = pos_tag(tokens)
        wordnet_input = get_wordnet_pos(tags)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t,tag) for t,tag in wordnet_input]
    # each token was converted to a string in the following line to fix an error that each word was being saved a np object
    tokens = [str(x) for x in tokens]
    return tokens
    

def lemmatize_row(df):
    '''
    Returns: df with lemmatized version of 'text' column in 'tokens' column
    
    Dependency: requires lemmatize_text and get_wordnet_pos functions
    '''
    try: 
        df['tokens'] = df['text'].apply(lemmatize_text) 

        # adding column with a string version of the 
        df['text_of_tokens'] = df['tokens'].apply(lambda x: ' '.join(x))
           
    except: 
        print("ERROR: Could not lemmatize text. See function lemmatize_text in data_cleaning.py to fix bug.")
    return df
 
    
# Main
def main():

    df = clean_df(RAW_DATA_FILE_PATH)
    print('SUCCESS: pickle file loaded and df cleaned')
    
    df = lemmatize_row(df)
    print('SUCCESS: lemmatized text added.')
    
    # Explort final dataframe to pkl file in data folder (or provided directory) 
    try:   
        with open(CLEAN_DATA_FILE_PATH, 'wb') as f:
            pickle.dump(df, f)
        print(f'SUCCESS: Saved dataframe to {CLEAN_DATA_FILE_PATH}!')
    except: 
        print(f"ERROR: Could not save dataframe to {CLEAN_DATA_FILE_PATH}.")

    return None


if __name__ == '__main__':
    main()