
# import numpy
# import torch
# import jobliB
import os
import pandas as pd

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
model_dir = os.path.join(os.getcwd(), 'models', 'bert_model')

embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
topic_model = BERTopic.load(model_dir, embedding_model=embedding_model)
topics_found_series = topic_model.get_topic_info()['Topic']
topics_mapping_dict = {topics_found_series[i]: 'topic_'+ str(i) for i in range(len(topics_found_series))}

print()
print("******************************************** Model Loaded Successfully **********************************")
print()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/lda')
def lda_result():
    # Render the LDAvis HTML page
    return render_template('lda_vis_data_gensim_models.html')


# USER SEND
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

        # Read the file
        with open(file_path, 'r') as f:
            data = f.read()

        # Read the CSV file with no delimiter (everything treated as a single column)
        #with open('test.csv', 'r') as f:
        #    data = f.readlines()

        # Clean the data: remove newlines, commas, and quotes
        new_documents = [line.strip().replace(',', '').replace('"', '') for line in data]

        # Read into a Pandas DataFrame
        # df = pd.read_csv(file, header=None, quotechar='"', names=["text"])
        df = pd.DataFrame(new_documents, columns=['text'])

        # Predict topics for new documents
        # print("**********************************", new_documents, "****************************")
        new_topics, new_probabilities = "No PREDICT done YET", "Not CALCULATED YET"
        new_topics, new_probabilities = topic_model.transform(new_documents)

        df['topics'] = new_topics
        df['probabilities'] = new_probabilities
        df['probabilities'] = df['probabilities'].apply(lambda x: f"{x:.0%}")
        df['topics'] = df['topics'].map(topics_mapping_dict)

        # Display BOTH the content of the CSV as HTML table AND the TOPICS FOUND
        return df.head().to_html()

    return "Invalid file type. Only CSV files are allowed.", 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
    #gunicorn -w 4 -b 0.0.0.0:5000 main:app

