''''
Description:
- This script takes for input a csv of restaurant reviews.
- The reviews are cleaned and a topic is detected with a probability for each review. 
- The resulting dataframe is saved as a csv.
- The output is used for the EAT HERE app


Author: Ashley ACKER
'''

# Imports & Constants
import pickle
import os
import pandas as pd
import re
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import joblib
import hdbscan 


# Cluster_names will be modified by us once we decide on the topics we want to use
OUTPUT_FOLDER = "data_processing_outputs"
CLUSTER_NAMES = {0: 'Topic_A', 1: 'Topic_B'}


def load_data(FILE_PATH_DATA = os.path.join('data', 'yelp_data_cleaned.pkl')):
    with open(FILE_PATH_DATA, 'rb') as f:
        df = pickle.load(f)
    return df


def add_clusters(df):
    
    FILE_PATH_CLUSTER_LABELS = os.path.join('HDBSCAN_model', 'cluster_labels.pkl') 
    with open(FILE_PATH_CLUSTER_LABELS, 'rb') as f:
        cluster_labels = pickle.load(f)

    FILE_PATH_PROBABILITIES = os.path.join('HDBSCAN_model', 'probabilities.pkl') 
    with open(FILE_PATH_PROBABILITIES, 'rb') as f:
        probabilities = pickle.load(f)

    df['cluster_labels'] = cluster_labels
    df['probabilities'] = probabilities

    # Remove noise
    #df = df[df['cluster_labels']!=-1]
    return df


def create_top10_per_cluster_csv(df):
    grouped_df = df.sort_values(by=['cluster_labels', 'probabilities'], ascending=[True, False])

    # Select top 10 rows per cluster based on probabilities
    top10_per_cluster = grouped_df.groupby('cluster_labels').head(10)

    # Select relevant columns (text, cluster_labels, probabilities)
    top10_per_cluster = top10_per_cluster[['cluster_labels', 'probabilities', 'text']]

    # Save to CSV
    top10_per_cluster.to_csv(os.path.join(OUTPUT_FOLDER, 'top_10_per_cluster.csv'), index=False, header=False)
    
    return None


def add_cluster_names(df): 
    df['CLUSTER_NAMES'] = df['cluster_labels'].map(CLUSTER_NAMES)
    df.to_csv(os.path.join(OUTPUT_FOLDER,"df_with_topics.csv"), header=True)
    return None


def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()


def create_BERT_embedding(reviews, model_name = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(reviews, show_progress_bar=True)
    return embeddings


def load_and_process_new_data(FILE_PATH = 'new_data.csv'):

    new_df = pd.read_csv(FILE_PATH, header=None)
    new_df = new_df.rename(columns={0: 'text'}).drop(1, axis=1)
    new_df['cleaned_text'] = new_df['text'].apply(clean_text)

    embeddings = create_BERT_embedding(new_df['cleaned_text'].tolist())
    
    PCA_components = 5 # need to find this number in the log.txt file from HDBSCAN_model.py
    pca = PCA(n_components=PCA_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    # Load the saved HDBSCAN model
    FILE_PATH_HDBSCAN = os.path.join('model_outputs', 'hdbscan_model.joblib') 
    hdbscan_model = joblib.load(FILE_PATH_HDBSCAN)

    # Predict clusters for new data
    cluster_labels, probabilities = hdbscan.approximate_predict(hdbscan_model, embeddings_pca)

    # Add cluster labels and probabilities to new_df
    new_df['cluster_labels'] = cluster_labels
    new_df['probabilities'] = probabilities
    new_df = new_df.sort_values(by=['cluster_labels', 'probabilities'], ascending=[True, False])
    
    # Remove noise
    new_df_without_noise = new_df[new_df['cluster_labels']!=-1]
    
    # Add cluster names
    new_df_without_noise['CLUSTER_NAMES'] = new_df_without_noise['cluster_labels'].map(CLUSTER_NAMES)
    
    # Change column order
    new_df_without_noise = new_df_without_noise[['text','CLUSTER_NAMES', 'probabilities' ]]

    # Save to CSV
    new_df_without_noise.to_csv(os.path.join(OUTPUT_FOLDER,"new_df_with_topics.csv"), header=True, index=False)

    return None


def main():
    
    # PART 1: ADD CLUSTERS AND CREATE TOP 10 PER CLUSTER CSV
    df = load_data()
    df = add_clusters(df)
    df.to_csv(os.path.join(OUTPUT_FOLDER,"df_with_clusters.csv"), header=True, index=False)
    print("\nSUCCESS: df_with_clusters.csv created.\n")
    
    create_top10_per_cluster_csv(df)
    print("\nSUCCESS: top 10 reviews per cluster csv created.\n")
    
    print(
    '''\nTO DO: 
    - Analyze top_10_per_cluster.csv to identify the topic names.  
    - Update CLUSTER_NAMES accordingly in script.
    - Update PCA_components in load_and_process_new_data according to model selected (see log.txt) and then continue to execute PART 2. \n
    ''')
    
    # PART 2: PROCESS NEW DATA AND ADD CLUSTER NAMES
    # load_and_process_new_data()  
    # print("\nSUCCESS: new data processed and cluster names added. Results saved to new_df_with_topics.csv.\n")

    return None


if __name__ == '__main__':
    main()