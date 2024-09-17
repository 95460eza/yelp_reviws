
import os
import csv
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask, request, render_template, redirect, url_for

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

# Load the saved BERTopic model
# model_path = os.path.join( os.getcwd(), 'models', 'model_all_MiniLM_L12_v2.pkl')
# model_path = os.path.join( os.getcwd(), 'models', 'model_all_MiniLM_L12_v2_joblib.joblib')
# model_path = os.path.join( os.getcwd(), 'models', 'model_all_MiniLM_L12_v2_torch.pth')


# topic_model = BERTopic.load(model_path)
# map_location=torch.device('cpu')
# topic_model = torch.load(model_path, map_location=map_location)
# print("Model Loaded")

# Load the BERTopic model with PyTorch-based embedding
#model_dir = os.path.join(os.getcwd(), 'models', 'bert_model')
model_dir = os.path.join(os.getcwd(), 'models', 'bertopic_500')

embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
topic_model = BERTopic.load(model_dir, embedding_model=embedding_model)
#topics_found_series = topic_model.get_topic_info()['Topic']
# topics_found_series = topic_model.get_topic_info()['Topic']
#human_labeled_topics = ['Food quality', 'Hygiene','Stopping serving before time','Hotel Complaints','Order Wrong']
#human_labeled_topics = ['outliers', 'Long wait time for deliveries/food to go','Long wait time for deliveries/food to go',
#                       'Not hygienic enough (e.g. no masks during pandemic, dirty establishment)',
#                        'Issue related to services outside restauration (e.g. lodging)',
#                        'Too strict on hours of service', 'Not hygienic enough (e.g. no masks during pandemic, dirty establishment)',
#                        'Not hygienic enough (e.g. no masks during pandemic, dirty establishment)',  'Lacking customer service',
#                        'Prejudice servers (e.g. racist remarks)', 'Prejudice servers (e.g. racist remarks)',
#                       'Lacking customer service'
#                        ]

#topics_mapping_dict = {topics_found_series[i]: 'topic_'+ str(i) for i in range(len(topics_found_series))}
#topics_mapping_dict = {topics_found_series[i]: human_labeled_topics[i] for i in range(len(topics_found_series))}
topics_mapping_dict = {
                 -1: 'outlier',
                 0: 'Long wait time for deliveries/food to go',
                 1: 'Long wait time for deliveries/food to go',
                 2: 'Not hygienic enough (e.g. no masks during pandemic, dirty establishment)',
                 3: 'Issue related to services outside restauration (e.g. lodging)',
                 4: 'Too strict on hours of service',
                 5: 'Not hygienic enough (e.g. no masks during pandemic, dirty establishment)',
                 6: 'Lacking customer service',
                 7: 'Prejudice servers (e.g. racist remarks)',
                 8: 'Lacking customer service',
                 }

#print("\n******************************************** Model Loaded Successfully **********************************\n")
#print("\n*********************************************************************************************************\n") 



# Extract topics for each cluster using CountVectorizer or TF-IDF Vectorizer
def extract_topics(documents, top_n=6):
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(documents)
    
    word_freq = np.array(X.sum(axis=0)).flatten()
    words = np.array(vectorizer.get_feature_names_out())
    
    top_words = list(words[word_freq.argsort()[::-1][:top_n]])
    
    top_words_freq= list(word_freq[word_freq.argsort()[::-1][:top_n]])
    top_words_freq = [int(x) for x in top_words_freq]
    return top_words,top_words_freq


@app.template_filter('zip')
def zip_lists(a, b):
    return zip(a, b)

      
@app.route('/')
def index():
    return render_template('index.html')



# USER SENDS HIS FILE
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file and file.filename.endswith('.csv'):
        # Save the file temporarily
        file_path = 'temp.csv'
        file.save(file_path)

        # Read the CSV file with no delimiter (everything treated as a single column)
        # with open('test.csv', 'r') as f:
        #    data = f.read()


        # Display BOTH the content of the CSV as HTML table AND the TOPICS FOUND
        #return df.head().to_html()
        return redirect(url_for('show_data', filename=file.filename))

    return "Invalid file type. Only CSV files are allowed.", 400



@app.route('/show_data/<filename>')
def show_data(filename):
    
    sentences = []
    
    file_path = 'temp.csv'
    
    # Read the CSV file with no delimiter (everything treated as a single column)
    
        
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Ensure that the row is not empty
                sentences.append(row[0])

    # Clean the data: remove newlines, commas, and quotes
    new_documents = [line.strip().replace(',', '').replace('"', '') for line in sentences]

    # Read into a Pandas DataFrame
    # df = pd.read_csv(file, header=None, quotechar='"', names=["text"])
    df = pd.DataFrame(new_documents, columns=['Review Analyzed'])
    
    # Predict topics for new documents
    
    new_topics, new_probabilities = "No PREDICT done YET", "Not CALCULATED YET"
    new_topics, new_probabilities = topic_model.transform(new_documents)
    
    
    df['Topic'] = new_topics
    df['Topic2'] = df['Topic'] 
    
    # EMPTY DICTIONARY WITH EACH KEY BEING A CLUSTER ID. VALUES ARE LIST OBJECTS
    unique_clusters = np.unique(topic_model.get_topic_info()['Topic'])
    clustered_documents = {cluster_number: [] for cluster_number in unique_clusters}
    for index_in_dataframe_for_doc, cluster_number_for_doc in enumerate(df['Topic']):
        #print(index_in_dataframe_for_doc, cluster_number_for_doc)
        clustered_documents[cluster_number_for_doc].append( df['Review Analyzed'][index_in_dataframe_for_doc] )
    
    for key, values in clustered_documents.items():
        print(f'********************************{key}*******{type(values)}**{len(values)}***********************************') 
        
    df['Topic'] = df['Topic'].map(topics_mapping_dict)
    
    
    df['Proba in Review'] = new_probabilities
    df['Proba in Review'] = df['Proba in Review'].apply(lambda x: f"{x:.0%}")
   
    
    df['New Text'] = df['Review Analyzed']
    df['Review Analyzed'] =  df['Review Analyzed'].apply(lambda x: x[0:80])
    
        
    
    #*********PREPARE SUMMARY DATA TABLE FOR THE NEW REVIEWS **************    
            
    # Group by 'topic' and count the number of rows in each group
    
    mask_outliers_in = df['Topic']=='outlier'
    df2 = df[~mask_outliers_in]  
    grouped_counts= df2.groupby('Topic').agg(
                                            count_col1=('Topic', 'size'),   # Count the number of rows in each group of col1
                                            max_col2=('Topic2', 'max')       # Get the maximum value of col2 in each group
                                            ).reset_index().sort_values(by='count_col1', ascending=False)  

    grouped_counts = grouped_counts.rename(columns={'count_col1': 'Number Reviews'})    
    grouped_counts['max_col2'] = grouped_counts['max_col2'].map(clustered_documents)   
    grouped_counts['max_col2'] = grouped_counts['max_col2'].apply(lambda x: extract_topics(x))   
    
        
    group_total = grouped_counts['Number Reviews'].sum()
    grouped_counts['Proportion'] = grouped_counts['Number Reviews']/group_total
    grouped_counts['Proportion'] = grouped_counts['Proportion'].apply(lambda x: f"{x:.0%}")
    
    
    grouped_counts_columns = list(grouped_counts.columns)
    
    #data = grouped_counts.to_html(classes='table table-striped', header="true", index=False)        
    data = grouped_counts.values.tolist()
    #print(data) 
    


    
    #*********PREPARE INDIVIDUAL REVIEWS TABLE**************
    df_to_use = df[~mask_outliers_in]
    df_to_use = df_to_use[['Review Analyzed','Topic','Proba in Review']]
    df_to_use = df_to_use.sort_values(by=['Topic', 'Proba in Review'], ascending=[False, False])
    columns = list(df_to_use.columns)
    # Convert DataFrame to list of lists
    #data2 = df.to_html(classes='table table-striped', header="true", index=False) 
    data2 = df_to_use.values.tolist()
    
    
    
    data3 = df[['New Text']].values.tolist()
    #print(df.shape, df[['New Text']].iloc[0])
    
    #*********PREPARE DATA FOR PIE CHART**************
    
    #pie_data = grouped_counts['Topic'].value_counts().reset_index()
    #pie_data.columns = ['label', 'proportion']   
    grouped_counts_us = pd.read_csv('grouped_counts_us.csv')   
    mask_outlier_us = grouped_counts_us['Topic']== 'outlier' 
    grouped_counts_us = grouped_counts_us[~mask_outlier_us]
    # Calculate the total
    total = grouped_counts_us['Number of Reviews'].sum()
    grouped_counts_us['proportion'] = grouped_counts_us['Number of Reviews'] / total
    #grouped_counts['proportion'] = grouped_counts['proportion'].apply(lambda x: f"{x:.0%}")


    chart_data = {
        'labels': grouped_counts_us['Topic'].tolist(),
        'data': grouped_counts_us['proportion'].tolist()
    }

    #return render_template('show_data.html', table=data)
    #return render_template('show_data.html', table=data, table2=data2, chart_data=chart_data, columns = columns)
    return render_template('show_data.html', table=data, table2=data2, table3=data3, grouped_counts_columns = grouped_counts_columns,
                                             chart_data=chart_data, columns = columns)
    #return render_template('show_data.html', table2=data2, chart_data=chart_data)



@app.route('/review/<string:review>')
def display_review(review):
    filename = 'temp.csv'
    # Here, you can retrieve the full review content from the database or pass it directly
    return render_template('review.html', review=review,  filename=filename)


@app.route('/main_words/<string:main_words_used1>/<string:main_words_used2>')
def main_words(main_words_used1, main_words_used2):
    filename = 'temp.csv'
    # Here, you can retrieve the full review content from the database or pass it directly
    return render_template('main_words.html', main_words_used1=main_words_used1, main_words_used2=main_words_used2, filename=filename)


@app.route('/lda')
def lda_result():
    # Render the LDAvis HTML page
    return render_template('lda_vis_data_gensim_models.html')


# If your DOCKERFILE has: "ENTRYPOINT gunicorn --bind 0.0.0.0:5000 main:app", the CONTAINER WILL LISTEN port 5000 AND the Flask development server will be bypassed (port=8000)
#  you cannot use gunicorn directly WITHIN the main.py file B/C gunicorn is a command-line tool and NOT a Python FUNCTION!!!!!
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
    #gunicorn -w 4 -b 0.0.0.0:5000 main:app

