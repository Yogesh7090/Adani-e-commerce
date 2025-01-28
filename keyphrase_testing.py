import spacy
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
# Initializing the spaCy model instance
nlp = spacy.load('en_core_web_sm')
 
df = pd.read_excel('People Analytics_POC Plan 1.xlsx', sheet_name='All Reviews')
 
keywords_whole_list = []
 
def get_keyphrases(input_text):
    keywords = []
    spacy_doc = nlp(input_text)
    # Extracting keyphrases
    for chunk in spacy_doc.noun_chunks:
        if chunk.text.lower() not in nlp.Defaults.stop_words:
            keywords.append(chunk.text)
    return  ",".join(keywords)
 
df['keyphrases'] = df['Review'].apply(get_keyphrases)
 
df.to_excel('output.xlsx')