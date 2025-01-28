#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:59:24 2024

@author: samridhishukla
"""

import nltk
import pandas as pd
import re
from rake_nltk import Rake
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


#----------------------------------------------------------------------------------------------------------------------------

df=spark.read.format("csv").option("header","true").load("/mnt/forecastdata/Review Data/landmarkgroupsplashfashion.csv")

df=df.toPandas()
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df=df[['content','at','score']]
df.rename(columns={'content': 'Review', 'at': 'Date','score':'Rating'}, inplace=True)
# Define column names for various data fields
review_column = 'Review'  # Column containing review text 
keyphrases_column = 'Keyphrases'  # Column to store extracted keyphrases  
cleaned_keyphrases_column = 'Cleaned_Keyphrases'  # Column to store cleaned and filtered keyphrases
document_id_column = 'document_id'  # Primary key column for identifying documents

# Parameters for keyphrase extraction
num_phrases = 3  # Number of keyphrases to extract
min_length = 2   # Minimum length of a keyphrase
max_length = 5   # Maximum length of a keyphrase

# Download necessary resources for Natural Language Processing with NLTK
nltk.download('stopwords')  # Download stopwords data
nltk.download('punkt')      # Download punkt tokenizer data


# Define a function to extract keyphrases using Rake algorithm
def extract_keyphrases_rake(text, num_phrases=num_phrases):
    rake = Rake()  # Initialize Rake object
    rake.extract_keywords_from_text(text)  # Extract keywords from the text
    keyphrases = rake.get_ranked_phrases()[:num_phrases]  # Get the top N ranked keyphrases  
    return ', '.join(keyphrases)  # Return the extracted keyphrases as a comma-separated string

# Define a function to clean a keyphrase
def clean_phrase(phrase):
    phrase = re.sub(r"[^\w\s]", '', phrase)  # Remove punctuation from the phrase
    phrase = re.sub(r"\b\d+\b", '', phrase)  # Remove any standalone numbers from the phrase  
    phrase = phrase.lower().strip()  # Convert the phrase to lowercase and remove leading/trailing whitespaces        
    return phrase

# Define a function to extract and clean keyphrases
def extract_and_clean_keyphrases(text, num_phrases=num_phrases, min_length=min_length, max_length=max_length):
    rake = Rake()  # Initialize Rake object
    rake.extract_keywords_from_text(text)  # Extract keywords from the text
    keyphrases = rake.get_ranked_phrases()  # Get all ranked keyphrases

    # Clean and filter phrases based on length
    cleaned_phrases = [clean_phrase(phrase) for phrase in keyphrases]  # Clean each keyphrase
    meaningful_phrases = [phrase for phrase in cleaned_phrases if min_length <= len(phrase.split()) <= max_length]  # Filter phrases by length

    return ', '.join(meaningful_phrases[:num_phrases])  # Return the top N cleaned and meaningful keyphrases as a string

# Add a document ID column to the DataFrame for identification
df[document_id_column] = range(1, len(df) + 1)

# Extract keyphrases using Rake and store them in the DataFrame
df[keyphrases_column] = df[review_column].apply(lambda x: extract_keyphrases_rake(str(x)))

# Extract and clean keyphrases, then store them in the cleaned keyphrases column
df[cleaned_keyphrases_column] = df[keyphrases_column].apply(lambda x: extract_and_clean_keyphrases(str(x)))

output_data = []
for idx, row in df.iterrows():
    review = row[review_column]
    keyphrases = row[cleaned_keyphrases_column].split(', ')
    for keyphrase in keyphrases:
        output_data.append({'Review': review, 'Keyphrase': keyphrase})
output_df = pd.DataFrame(output_data)


#----------------------------------------------------------------Next Step-------------------------------------------------------------


df=spark.read.format("csv").option("header","true").load("/mnt/forecastdata/Review Data/landmarkgroupsplashfashion.csv")
df=df.toPandas()
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df=df[['content','at','score']]
df.rename(columns={'content': 'Review', 'at': 'Date','score':'Rating'}, inplace=True)
expected_column = 'Review'  # Specify the column in the DataFrame containing the text to be analyzed for sentiment.
sentiment_column = 'Sentiment'  # Define the name of the column to store the sentiment polarity (positive, negative, neutral).
scores_column = 'Scores'  # column to store the sentiment scores
pos_score_column = 'pos_score'  # column to store the positive sentiment score
neu_score_column = 'neu_score'  # column to store the neutral sentiment score 
neg_score_column = 'neg_score'  # column to store the negative sentiment score 
document_id_column = 'Document ID'  # Define the name of the primary key column
threshold = 0.1  # Define the threshold for classifying sentiments as positive or negative

# Download the VADER lexicon, a pre-trained model for sentiment analysis.
nltk.download('vader_lexicon')
 
# Initialize an instance of the SentimentIntensityAnalyzer for sentiment analysis.
sid = SentimentIntensityAnalyzer()
# Define a function to analyze the sentiment of a given text
def analyze_sentiment(text):
    scores = sid.polarity_scores(str(text))  # Calculate sentiment scores using the SentimentIntensityAnalyzer
    sentiment = 'NEUTRAL'  # Initialize sentiment as neutral
    # If the positive score exceeds the threshold and is greater than the negative score, classify as positive
    if scores['pos'] > threshold and scores['pos'] > scores['neg']:
        sentiment = 'POSITIVE'
    # If the negative score exceeds the threshold and is greater than the positive score, classify as negative
    elif scores['neg'] > threshold and scores['neg'] > scores['pos']:
        sentiment = 'NEGATIVE'
    return sentiment, scores

# Check if the expected column exists in the DataFrame
if expected_column not in df.columns:
    print(f"Column '{expected_column}' not found in the DataFrame.")
else:
    df[document_id_column] = range(1, len(df) + 1)  # Generate a sequential document ID as a primary key
    # Apply the analyze_sentiment function to the expected column and store sentiment and scores
    df[sentiment_column], df[scores_column] = zip(*df[expected_column].map(analyze_sentiment))
    # Extract and store individual sentiment scores in separate columns
    df[pos_score_column] = df[scores_column].apply(lambda x: x['pos'])
    df[neu_score_column] = df[scores_column].apply(lambda x: x['neu'])
    df[neg_score_column] = df[scores_column].apply(lambda x: x['neg'])
    # Drop the combined scores column
    df.drop(columns=[scores_column], inplace=True)
b=df.columns.to_list()
mod = [i.rstrip() for i in b]
mod = [i.replace(' ','_') for i in mod ]
mod_dict= dict(zip(b,mod))
df.rename(columns= mod_dict,inplace=True)
combined_df = pd.merge(output_df, df, on='Review') 
combined_df = combined_df.dropna(subset=['Date'])







#---------------------------------------------storing to SQL--------------------------------------------------------


spark = SparkSession.builder.appName("AzureSQLConnector").getOrCreate()

sdf = spark.createDataFrame(combined_df)

jdbc_url = "jdbc:sqlserver://catalytics-dw01.database.windows.net:1433;database=dunnhumby"

#connection properties
connection_properties = {
  "user" : "sqladmin",
  "password" : "7yZ63d2KdkAY",
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}


table_name = "Keyphrase_Data"


sdf.write.jdbc(
    url=jdbc_url,
    table=table_name,
    mode="overwrite",  # Change to "append" if you want to append the data to an existing table
    properties=connection_properties
)

# Stop the Spark session (optional, depending on your use case)
# spark.stop()
