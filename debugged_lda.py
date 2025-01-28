import pandas as pd
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import spacy
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset (adjusted based on your CSV structure)
input_file = r"D:\adani_poc\pre_processing\People Analytics_POC Plan 1.xlsx"  # Replace with your input file path
df = pd.read_excel(input_file, sheet_name='All Reviews')

nlp = spacy.load('en_core_web_sm')

df_keyphrase = pd.read_excel('People Analytics_POC Plan 1.xlsx', sheet_name='All Reviews')

keywords_whole_list = []

def get_keyphrases(input_text):
    keywords = []
    spacy_doc = nlp(input_text)
    # Extracting keyphrases
    for chunk in spacy_doc.noun_chunks:
        if chunk.text.lower() not in nlp.Defaults.stop_words:
            keywords.append(chunk.text)
    return ",".join(keywords)

df_keyphrase['keyphrases'] = df_keyphrase['Review'].apply(get_keyphrases)
print(df_keyphrase['keyphrases'])

# Preprocess Reviews
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Tokenize, remove stop words, and lowercase."""
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

print("Preprocessing text...")
df['processed_content'] = df['Review'].apply(preprocess_text)

# Prepare for LDA
texts = df['processed_content'].tolist()
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print("Training LDA model...")
num_topics = 5
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)

# Assign Dominant Topics and Keywords
def get_dominant_topic_and_keywords(lda_model, corpus_row):
    """Get the dominant topic and its keywords for a single row of the corpus."""
    topic_probs = lda_model[corpus_row]
    if not topic_probs:
        return np.nan, 0.0, ""
    dominant_topic, perc_contrib = max(topic_probs, key=lambda x: x[1])
    keywords = ", ".join([word for word, prob in lda_model.show_topic(dominant_topic)])
    return dominant_topic, perc_contrib, keywords

print("Assigning dominant topics and keywords...")
df[['Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords']] = df.apply(
    lambda row: pd.Series(get_dominant_topic_and_keywords(lda_model, corpus[row.name])), axis=1
)

# Observed Keywords Mapping (Manually define keyphrases and their topic names)
observed_keywords_mapping = {
    'work-life balance': 'Work-Life Balance',
    'salary': 'Salary and Benefits',
    'benefits': 'Salary and Benefits',
    'job security': 'Job Security',
    'promotion': 'Promotions, Appraisal and Advancement',
    'advancement': 'Promotions, Appraisal and Advancement',
    'company culture': 'Company Culture and Management',
    'management': 'Company Culture and Management',
    'skill development': 'Skill Development and Work Satisfaction',
    'work satisfaction': 'Skill Development and Work Satisfaction',
}

# Fallback topic name in case no specific mapping matches
fallback_topic_name = "General Issues"

def map_keywords_to_topic_name(keywords, mapping, fallback):
    """Map topic keywords to a predefined topic name."""
    for keyword in keywords.split(', '):
        for observed_keyword, topic_name in mapping.items():
            if observed_keyword in keyword.lower():  # Case-insensitive matching
                return topic_name
    return fallback

print("Mapping keywords to topic names...")
df['Topic_Name'] = df['Topic_Keywords'].apply(
    lambda x: map_keywords_to_topic_name(x, observed_keywords_mapping, fallback_topic_name)
)

# Create Document_ID column
df['Document_ID'] = range(1, len(df) + 1)

# Map columns for output
df['rating'] = df['Rating']  # Map Rating to rating
output_columns = [
    'Title',            # Title of the review
    'Rating',           # Rating given by the reviewer
    'Document_ID',      # Unique document identifier
    'Dominant_Topic',   # ID of the most prominent topic
    'Topic_Perc_Contrib', # Percentage contribution of the dominant topic
    'Topic_Name',       # Human-readable topic name
    'Topic_Keywords'    # Keywords for the dominant topic
]

# Validate columns
available_columns = [col for col in output_columns if col in df.columns]

if len(available_columns) < len(output_columns):
    print("Warning: Some columns were not found in the DataFrame and will be skipped:", 
          set(output_columns) - set(available_columns))

# Save the results to a CSV file
# output_file = r"D:\adani_poc\pre_processing\output_file_adani.csv"
# df[available_columns].to_csv(output_file, index=False, header=[
#     'Title',            # Header for 'Title'
#     'Rating',           # Header for 'Rating'
#     'Document_ID',      # Header for 'Document_ID'
#     'Dominant_Topic',   # Header for 'Dominant_Topic'
#     'Topic_Perc_Contrib', # Header for 'Topic_Perc_Contrib'
#     'Topic_Name',       # Header for 'Topic_Name'
#     'Topic_Keywords'    # Header for 'Topic_Keywords'
# ])
# # print(f"Results saved to {output_file}")
