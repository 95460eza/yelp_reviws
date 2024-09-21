
'''
Description: 
- loads the cleaned yelp reviews from a pickle file
- creates the BERT embeddings
- perofrms PCA on embedding
- creates the HDBSCAN clustering model based on PCA using grid search to find the best parameters
- used to create models for the resto app EAT HERE

Author: Ashley ACKER

'''

# Imports:
import pickle
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import hdbscan
import pandas as pd
import os
from sklearn.metrics import silhouette_score
import itertools
import joblib
import numpy as np
from datetime import datetime 


# Constants:
DATA_DIR = "data"
MODEL_DIR = "HDBSCAN_model"
SAMPLE_SIZE = 242467
FILE_DIR = os.path.join(DATA_DIR, "yelp_data_cleaned.pkl")


#### Functions:

def load_data(file = FILE_DIR, sample_=True) -> pd.DataFrame:
    with open(file, 'rb') as f:
        df = pickle.load(f)
    if sample_: 
        df = df.sample(SAMPLE_SIZE)
    return df 


def create_param_grid(): 
    param_grid = {
        'pca__n_components': [10, 25, 50, 100, 150],  # Number of components for PCA. We tested and 10 components corresponds to about 30% of the variance, so that's our minimum. 150 == 87% of variance
        'hdbscan__min_cluster_size':  [int(SAMPLE_SIZE*0.0001), int(SAMPLE_SIZE*0.0003), int(SAMPLE_SIZE*0.001)], #the minimum size a final cluster can be. The higher this is, the bigger your clusters will be 
        'hdbscan__min_samples': [5, 50, 100, 150, 200],  # the minimum number of neighbours to a core point. The higher this is, the more points are going to be discarded as noise/outliers
        'hdbscan__metric': ['euclidean', 'manhattan'],  # Distance metric for HDBSCAN
    }
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        param_grid['pca__n_components'],
        param_grid['hdbscan__min_cluster_size'],
        param_grid['hdbscan__min_samples'],
        param_grid['hdbscan__metric']
    ))
    
    return param_combinations


def create_BERT_embedding(reviews, model_name = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(reviews, show_progress_bar=True)
    return embeddings


def HDBSCAN_grid_search(embeddings, param_combinations):
    
    best_model_parameters = {
        'pca__n_components': [] ,
        'hdbscan__min_cluster_size': [] ,
        'hdbscan__min_samples': [],
        'hdbscan__metric':[],
        'score': []
    }
    
    score_compare = -1
    i = 31
    
    for params in param_combinations[31:]:
 
        pca__n_components, hdbscan__min_cluster_size, hdbscan__min_samples, hdbscan__metric = params
        
        pca = PCA(n_components=pca__n_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan__min_cluster_size, 
                                    min_samples=hdbscan__min_samples, 
                                    metric=hdbscan__metric, 
                                    prediction_data=True)
        
        cluster_labels = clusterer.fit_predict(embeddings_pca)
        
        # Filter out noise points
        mask = cluster_labels != -1
        embeddings_pca_filtered = embeddings_pca[mask]
        cluster_labels_filtered =  cluster_labels[mask] # np.random.randint(3, size=100)
        
        if (embeddings_pca_filtered.shape[0] == 0):
            log_results = f'iteration: {i} ---> ' + 'cluster_labels is only -1s. ' + str(params) + datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  '\n' +  '\n'
            with open("log.txt", "a") as file:
                file.write(log_results)
            
        else: 
            n_clusters = len(np.unique(cluster_labels_filtered))
            
            score = silhouette_score(embeddings_pca_filtered, cluster_labels_filtered, metric=hdbscan__metric)

            if score > score_compare:
                score_compare = score
                best_model_parameters = {
                    'pca__n_components': pca__n_components,
                    'hdbscan__min_cluster_size': hdbscan__min_cluster_size,
                    'hdbscan__min_samples': hdbscan__min_samples,
                    'hdbscan__metric': hdbscan__metric, 
                    'score': score
                }
                
            log_results = f'iteration: {i} ---> ' + f'Number of clusters: {n_clusters}, ' + f'Score: {score}, ' +  '\n    ' + 'Parameters: ' + str(params) +  '\n    ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + '\n' +  '\n'  
            with open("log.txt", "a") as file:
                file.write(log_results)
        i += 1
               
    return best_model_parameters
  
  
def HDBSCAN_clustering(embeddings, best_model_parameters):
    
    pca__n_components_best = best_model_parameters['pca__n_components']
    hdbscan__min_cluster_size_best = best_model_parameters['hdbscan__min_cluster_size']
    hdbscan__min_samples_best = best_model_parameters['hdbscan__min_samples']
    hdbscan__metric_best = best_model_parameters['hdbscan__metric']
    
    pca = PCA(n_components=pca__n_components_best, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    # Getting number of components to be able to prepare new data before transforming with HDBSCAN model
    with open("log.txt", "a") as file:
        file.write(f'Shape of PCA in final model: {embeddings_pca.shape}.\n')
    
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan__min_cluster_size_best,
                                min_samples=hdbscan__min_samples_best,
                                metric=hdbscan__metric_best,
                                prediction_data=True)
    
    cluster_labels = clusterer.fit_predict(embeddings_pca)
    
    return clusterer, cluster_labels, embeddings_pca  
    

def main():
    
    # Initiate log file and make sure it's empty before adding log information
    with open('log.txt', 'w') as file:
        file.write("HDBSCAN_model.py log file.\nStarted on: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ".\n")
    
    # df = load_data()
    # with open("log.txt", "a") as file:
    #     file.write('Data loaded.\n')
    
    param_combinations = create_param_grid()
    with open("log.txt", "a") as file:
        file.write('Param combinations created.\n')
    
    bert_name = f'bert_embeddings_{SAMPLE_SIZE}.pkl'
    embeddings = load_data(file = os.path.join(MODEL_DIR, bert_name), sample_=False)
    # embeddings = create_BERT_embedding(df['cleaned_text'].tolist())
    with open("log.txt", "a") as file:
        text = 'BERT embeddings complete. Shape of embeddings: ' + str(embeddings.shape) + '\n'
        file.write(text)
    # with open(os.path.join(MODEL_DIR, bert_name), 'wb') as f:
    #     pickle.dump(embeddings, f)
    # Below is an option to load the bert embeddings if they're already saved
    
    # best_model_parameters = HDBSCAN_grid_search(embeddings, param_combinations)
    # with open("log.txt", "a") as file:
    #     text =  'Grid search complete!\n' + f'Best model parameters: {best_model_parameters}\n'
    #     file.write(text)   
    
    # iteration: 38 ---> Number of clusters: 5, Score: 0.374734491109848, 
    # Parameters: (25, 24, 200, 'euclidean')
    # 12/09/2024 23:21:24
    
    best_model_parameters = {'pca__n_components': 25, 'hdbscan__min_cluster_size': 24, 'hdbscan__min_samples': 200, 'hdbscan__metric': 'euclidean'}
    clusterer, cluster_labels, embeddings_pca = HDBSCAN_clustering(embeddings, best_model_parameters)
    with open("log.txt", "a") as file:
        file.write('HDBSCAN model fitted with best parameters.\n')
        
    joblib.dump(clusterer, os.path.join(MODEL_DIR, 'hdbscan_model.joblib'))
    with open(os.path.join(MODEL_DIR, 'bert_embedding_pca.pkl'), 'wb') as f:
        pickle.dump(embeddings_pca, f)
    with open(os.path.join(MODEL_DIR, 'cluster_labels.pkl'), 'wb') as f:
        pickle.dump(cluster_labels, f)
    with open("log.txt", "a") as file:
        file.write('HDBSCAN model results saved.\n')

    probabilities = clusterer.probabilities_
    with open(os.path.join(MODEL_DIR, 'probabilities.pkl'), 'wb') as f:
        pickle.dump(probabilities, f)
    with open("log.txt", "a") as file:
        file.write('Probabilities saved.\n')

##################################################################################
    # best_model_parameters = HDBSCAN_grid_search(embeddings, param_combinations)
    # with open("log.txt", "a") as file:
    #     text =  'Grid search complete!\n' + f'Best model parameters: {best_model_parameters}\n'
    #     file.write(text)
    
    # clusterer, cluster_labels, embeddings_pca = HDBSCAN_clustering(embeddings, best_model_parameters)
    # with open("log.txt", "a") as file:
    #     file.write('HDBSCAN model fitted with best parameters.\n')
        
    # joblib.dump(clusterer, os.path.join(MODEL_DIR, 'hdbscan_model.joblib'))
    # with open(os.path.join(MODEL_DIR, 'bert_embedding_pca.pkl'), 'wb') as f:
    #     pickle.dump(embeddings_pca, f)
    # with open(os.path.join(MODEL_DIR, 'cluster_labels.pkl'), 'wb') as f:
    #     pickle.dump(cluster_labels, f)
    # with open("log.txt", "a") as file:
    #     file.write('HDBSCAN model results saved.\n')

    # probabilities = clusterer.probabilities_
    # with open(os.path.join(MODEL_DIR, 'probabilities.pkl'), 'wb') as f:
    #     pickle.dump(probabilities, f)
    # with open("log.txt", "a") as file:
    #     file.write('Probabilities saved.\n')


    return None


if __name__ == '__main__':
    main()
