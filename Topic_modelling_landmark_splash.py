#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:58:02 2024

@author: samridhishukla
"""

#Libraries
from azure.storage.blob import BlobServiceClient
import pandas as pd
import numpy as np 
import io
import pyspark.sql
import pyodbc
#from azure.storage.blob import BlockBlobService
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import pandas as pd
import numpy as np 
import io
import pyspark.sql
import pyodbc
#from azure.storage.blob import BlockBlobService
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from pyspark.sql import SparkSession


# Load the pandas library for data manipulation.
# Load the nltk library for natural language processing tasks.
# Import the SentimentIntensityAnalyzer class from nltk's vader module.
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Importing necessary libraries
import pandas as pd
import gensim
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models import CoherenceModel
import os

# Setting tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

############################ Mounting Process ################################
# mounting code

containerName = "flat-files"
storageAccountName = "demandforecastingblob"
 
config = "fs.azure.sas." + containerName+ "." + storageAccountName + ".blob.core.windows.net"
 
# # Provide the storage account key
storageAccountAccessKey="fkb4rRAi18AV542cnwtigP13nD6k35KdDboW5bpd67jUq8PCQuf/Dnb8cUemei8dhRnISLzla4pd+AStAlTFOg=="
 
dbutils.fs.mount(
   source = "wasbs://flat-files@demandforecastingblob.blob.core.windows.net",
   mount_point = "/mnt/forecastdata",
   extra_configs = {'fs.azure.account.key.' + storageAccountName + '.blob.core.windows.net': storageAccountAccessKey})

df=spark.read.format("csv").option("header","true").load("/mnt/forecastdata/Review Data/landmarkgroupsplashfashion.csv")
df=df.toPandas()
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df=df[['content','at','score']]
df.rename(columns={'content': 'Review', 'at': 'Date','score':'Rating'}, inplace=True)
review_column = 'Review'  # Column name containing reviews  
document_id_column = 'DocumentID'  # Column name containing document IDs  
no_below_threshold = 2  # Threshold for filtering out words appearing in fewer than 2 documents 
no_above_threshold = 0.15  # Threshold for filtering out words appearing in more than 25% of documents 
keep_n_words = 100000  # Maximum number of unique words to keep in the dictionary  
lda_passes = 2  # Number of passes (iterations) the LDA algorithm will make over the corpus
nltk.download('wordnet')  # Downloading WordNet data for lemmatization
nltk.download('omw-1.4')  # Downloading Open Multilingual WordNet data for additional language support

# Extracting text data and assigning document IDs
data_text = df[[review_column]]
data_text[document_id_column] = range(1, len(data_text) + 1)
lemmatizer = WordNetLemmatizer()  # Initializing WordNetLemmatizer instance for lemmatization

#Preprocessing Function
def preprocess(text):
    text = str(text).lower()  # Converting text to lowercase
    result = []  # Initializing empty list to store processed tokens
    for token in gensim.utils.simple_preprocess(text, deacc=True):  # Iterating over tokens after simple preprocessing
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:  # Filtering out stopwords and tokens with length less than 3
            result.append(lemmatizer.lemmatize(token, pos='v'))  # Lemmatizing tokens and adding to the result list
    return result  # Returning the list of processed tokens

# Applying preprocessing function to each review
processed_docs = data_text[review_column].map(preprocess)

# Creating dictionary and filtering extremes
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=no_below_threshold, no_above=no_above_threshold, keep_n=keep_n_words)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Defining range of topics to explore
topic_range = range(2, 15) 
best_model_metrics = {'coherence': 0, 'perplexity': float('inf'), 'num_topics': 0, 'model': None, 'topics': {}}
df[document_id_column] = range(1, len(df) + 1)  # Generate a sequential document ID as a primary key

# Iterating over different numbers of topics
for num_topics in topic_range:
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=lda_passes)  # Fitting LDA model
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')  # Calculating coherence score
    coherence_score = coherence_model_lda.get_coherence()  # Getting coherence score
    perplexity_score = lda_model.log_perplexity(bow_corpus)  # Calculating perplexity score
    
    # Updating best model based on higher coherence and lower perplexity
    if coherence_score > best_model_metrics['coherence'] and perplexity_score < best_model_metrics['perplexity']:
        best_model_metrics.update({'coherence': coherence_score, 'perplexity': perplexity_score, 'num_topics': num_topics, 'model': lda_model})
        best_model_metrics['topics'] = {i: [word for word, prob in lda_model.show_topic(i)] for i in range(num_topics)}
        
    # Printing metrics for each number of topics
    print(f"Num Topics: {num_topics}, Coherence Score: {coherence_score}, Perplexity: {perplexity_score}")


print(f"\nBest Model: {best_model_metrics['num_topics']} topics, Coherence Score: {best_model_metrics['coherence']}, Perplexity: {best_model_metrics['perplexity']}")
print("Topics of the Best Model:")
for topic_num, words in best_model_metrics['topics'].items():
    print(f"  Topic {topic_num + 1}: {words}")


def assign_topics_to_reviews(lda_model, corpus, data):
    topic_results = []
    for i, row_list in enumerate(lda_model[corpus]):
        row = sorted(row_list, key=lambda x: (x[1], -x[0]), reverse=True)
        dominant_topic = int(row[0][0])
        topic_perc_contrib = round(row[0][1], 4)
        
        # Retrieving top words for the dominant topic
        topic_keywords = ", ".join([word for word, prob in lda_model.show_topic(dominant_topic)])
        topic_results.append((data[document_id_column].iloc[i], dominant_topic, topic_perc_contrib, topic_keywords))
    return pd.DataFrame(topic_results, columns=[document_id_column, 'Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords'])

# Assigning topics to reviews and merging with original DataFrame
df_topic_keywords = assign_topics_to_reviews(best_model_metrics['model'], bow_corpus, data_text)
final_df = df.merge(df_topic_keywords, left_index=True, right_on=document_id_column).set_index(document_id_column)

# Function to assign topic names based on observed keywords
def assign_topic_names(topic_keywords):
    observed_keywords_mapping = {
    'app': 'Application Functionality',

    'service': 'Customer Service Experience',
    'delivery': 'Delivery and Logistics',
    'online': 'Online Shopping Experience',
    'order': 'Order Management',
    'customer': 'Customer Service Experience'
}
    
    keywords = topic_keywords.lower().split(', ')
    for keyword in keywords:
        for observed_keyword, topic_name in observed_keywords_mapping.items():
            if observed_keyword in keyword:
                return topic_name
    return 

# Applying topic names assignment function to final DataFrame
final_df['Topic_Name'] = final_df['Topic_Keywords'].apply(assign_topic_names)
b=final_df.columns.to_list()
mod = [i.rstrip() for i in b]
mod = [i.replace(' ','_') for i in mod ]
mod_dict= dict(zip(b,mod))
final_df.rename(columns= mod_dict,inplace=True)
def modify_dataframe(df, column_to_drop, column_to_rename, new_name):
    """
    Drop a column from the dataframe and rename another column if they exist.

    Parameters:
    df (pandas.DataFrame): The dataframe to modify.
    column_to_drop (str): The name of the column to drop, if it exists.
    column_to_rename (str): The name of the column to rename, if it exists.
    new_name (str): The new name for the column being renamed.

    Returns:
    pandas.DataFrame: The modified dataframe with the column dropped and renamed, if applicable.
    """
    # Create a copy of the dataframe to avoid modifying the original dataframe
    modified_df = df.copy()
    
    # Check if the column to drop exists before dropping it
    if column_to_drop in modified_df.columns:
        modified_df.drop(column_to_drop, axis=1, inplace=True)
    
    # Check if the column to rename exists before renaming it
    if column_to_rename in modified_df.columns:
        modified_df.rename(columns={column_to_rename: new_name}, inplace=True)
    
    return modified_df

# Usage
# Assuming 'final_df' is your original dataframe
final_df_modified = modify_dataframe(final_df, 'DocumentID_y', 'DocumentID_x', 'Document_ID')

# Now 'final_df_modified' contains the modified dataframe
print(final_df_modified) 
final_df_modified = final_df_modified.dropna(subset=['Date','Review'])
final_df_modified  = final_df_modified .drop_duplicates(subset=['Review'], keep='first')
final_df_modified  = final_df_modified [~final_df_modified ['Review'].str.lower().isin(['good','Good'])]



spark = SparkSession.builder.appName("AzureSQLConnector").getOrCreate()

sdf = spark.createDataFrame(final_df_modified)

jdbc_url = "jdbc:sqlserver://catalytics-dw01.database.windows.net:1433;database=dunnhumby"

#connection properties
connection_properties = {
  "user" : "sqladmin",
  "password" : "7yZ63d2KdkAY",
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}


table_name = "Topic_data"


sdf.write.jdbc(
    url=jdbc_url,
    table=table_name,
    mode="overwrite",  # Change to "append" if you want to append the data to an existing table
    properties=connection_properties
)

# Stop the Spark session (optional, depending on your use case)
# spark.stop()
