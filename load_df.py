'''
Module Description: 
This module allows you to load the reviews and business yelp datasets. 
The json files are extracted from the tar file and saved in the data folder.
The data is loaded from the json files and then merged to create a dataframe of reviews that included business information.

A sample of 14% of the review data is selected randomly and stratified by star rating to facilliate data use
The final dataframe is saved as a pkl file within the data folder.

WARNING: the yelp_academic_dataset_review.json can take a long time to load.

Author: Ashley ACKER

TODO:
- Data must be downloaded first from https://www.kaggle.com/yelp-dataset/yelp-dataset
- The tar file should be saved to a data folder, or user should modify the constant DATA_DIR accordingly.

'''

# Downloads & Imports 
import tarfile
import os
import pandas as pd
import pickle
from langdetect import detect, LangDetectException


# Constants
DATA_DIR = "data"
DATA_INPUT_FILE_PATH = os.path.join(DATA_DIR, 'yelp_dataset.tar') 
DATA_OUTPUT_FILE_PATH = os.path.join(DATA_DIR, 'yelp_reviews_raw_df.pkl')


# Functions
def extract_data(file_path=DATA_INPUT_FILE_PATH, data_dir=DATA_DIR):
    ''' 
    Extracts data from tar file to jsons in data folder.
    
    Returns: None
    '''
    # Extract all contents to the specified directory
    try: 
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(path=data_dir)
        return None
    except: 
        print("ERROR: Could not extract data from tar file. See function extract_data() in data_cleaning.py to fix bug.")


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'
 

def create_df(data_dir=DATA_DIR):
    ''' 
    Reads in the json dataset files and write it out to a csv files.
    Then use the reviews.csv file to create a pandas dataframe and left join buisinesses.csv.
    
    Returns: pandas df of reviews which includes information on the businesses.
    '''
    # Read in data from json files
    try: 
        raw_df_business = pd.read_json(os.path.join(data_dir, 'yelp_academic_dataset_business.json') , lines=True)
        print('SUCCESS: business data loaded.')
        
        chunks = pd.read_json(os.path.join(data_dir, 'yelp_academic_dataset_review.json'), lines=True, chunksize=10000)
        raw_df_review = pd.concat([chunk for chunk in chunks])
        print('SUCCESS: review data loaded.')
        
    except: 
        print('ERROR: Could not read in data from json files. See function create_df() in data_cleaning.py to fix bug.')
    
    # Taking sample of 14% of reivews stratified by star rating
    raw_df_review = raw_df_review.groupby('stars', group_keys=False).apply(lambda x: x.sample(frac=0.25, random_state=42))
    
    # Dropping stars feature from business data to facilitate merge with review data (which also has star feature)
    raw_df_business_selected = raw_df_business.drop(['stars'], axis=1)
    
    # Adding business data to review df
    raw_df = raw_df_review.merge(raw_df_business_selected, how='left', on='business_id') 
    
    # selecting features of interest
    raw_df = raw_df[[ 'city', 'state', 'review_id', 'business_id', 'stars', 'text',  'name', 'address', 'latitude', 'longitude', 'review_count', 'categories']]

    # Detecting language and adding in column to df
    raw_df['language'] = raw_df['text'].apply(detect_language)
    print('SUCCESS: language detection completed.')

    return raw_df  


# Main
def main():
    
    # extract_data() # !!! Commented for now because we don't need to extract data again
    print('SUCCESS: Data extracted from tar file.')
    
    df = create_df() 
    
    try:  
        with open(DATA_OUTPUT_FILE_PATH, 'wb') as f:
            pickle.dump(df, f)
        print(f'SUCCESS: Raw yelp data on reviews and business saved to {DATA_OUTPUT_FILE_PATH}!')
        
    except: 
        print(f"ERROR: Could not save dataframe to {DATA_OUTPUT_FILE_PATH}.")

    return None


if __name__ == '__main__':
    main()