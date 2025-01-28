#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:56:17 2024

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



expected_column = 'Review'  
sentiment_column = 'Sentiment' 
scores_column = 'Scores'  
document_id_column = 'Document ID'
positive_threshold = 0.6
negative_threshold = -0.3
 
df=spark.read.format("csv").option("header","true").load("/mnt/forecastdata/Review Data/landmarkgroupsplashfashion.csv")

df=df.toPandas()
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df=df[['content','at','score']]
df.rename(columns={'content': 'Review', 'at': 'Date','score':'Rating'}, inplace=True)
nltk.download('vader_lexicon')

sentiment_analyzer = SentimentIntensityAnalyzer()
 
df[document_id_column] = range(1, len(df) + 1)

def calculate_sentiment(review):
    if review is None:
        # Handling None values by returning a neutral compound score
        return 0.0
    scores = sentiment_analyzer.polarity_scores(review)
    compound_score = scores['compound']
    return compound_score
 
df['Scores'] = df['Review'].apply(calculate_sentiment)

def classify_sentiment_threshold(score):
    if score >= positive_threshold:
        return 'Positive'
    elif score <= negative_threshold:
        return 'Negative'
    else:
        return 'Neutral'
 
df['Sentiment'] = df['Scores'].apply(classify_sentiment_threshold)
b=df.columns.to_list()
mod = [i.rstrip() for i in b]
mod = [i.replace(' ','_') for i in mod ]
mod_dict= dict(zip(b,mod))
df.rename(columns= mod_dict,inplace=True)
df['Brand'] = 'Splash'  
df['Company'] = 'LandmarkGroup'
df = df.dropna(subset=['Date','Review'])
df = df.drop_duplicates(subset=['Review'], keep='first')
df = df[~df['Review'].str.lower().isin(['good','Good'])]



spark = SparkSession.builder.appName("AzureSQLConnector").getOrCreate()

sdf = spark.createDataFrame(df)

jdbc_url = "jdbc:sqlserver://catalytics-dw01.database.windows.net:1433;database=dunnhumby"

#connection properties
connection_properties = {
  "user" : "sqladmin",
  "password" : "7yZ63d2KdkAY",
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

table_name = "Sentiment_Data"

sdf.write.jdbc(
    url=jdbc_url,
    table=table_name,
    mode="overwrite",  # Change to "append" if you want to append the data to an existing table
    properties=connection_properties
)

# Stop the Spark session (optional, depending on your use case)
# spark.stop()
