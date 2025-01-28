import spacy
import pandas as pd

from spacytextblob.spacytextblob import SpacyTextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initializing the spaCy model instance
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Load the input Excel file
input_file = 'People Analytics_POC_2.xlsx'

# Load the required sheet into DataFrame
df = pd.read_excel(input_file, sheet_name='All Reviews_Topic ')

keywords_whole_list = []

def get_keyphrases(input_text):
    keywords = []
    spacy_doc = nlp(input_text)
    # Extracting keyphrases
    for chunk in spacy_doc.noun_chunks:
        if chunk.text.lower() not in nlp.Defaults.stop_words:
            keywords.append(chunk.text)
    return ",".join(keywords)

def sentiment_anlysis(input_text):
    doc = nlp(input_text)
    sentiment = doc._.blob.polarity
    sentiment_score = round(sentiment, 2)

    return sentiment_score

def sentiment_label(sentiment_score):
    if sentiment_score > 0.2:
        sentiment_label = 'Positive'
    elif sentiment_score < -0.2:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    return sentiment_label

# Function to map keyphrases to predefined topics using similarity scoring
def map_keyphrases_to_topic_name_with_score(keyphrases, topic_names):
    """Map extracted keyphrases to predefined topics using cosine similarity."""
    # Convert keyphrases and topics into a TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    all_text = [keyphrases] + topic_names  # First element is the keyphrases
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)

    # Calculate cosine similarity between keyphrases and each topic
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Find the topic with the highest similarity score
    best_match_idx = similarity_scores.argmax()
    best_topic = topic_names[best_match_idx]
    best_score = similarity_scores[best_match_idx]

    return best_topic, best_score

topic_names = [
    "Work-Life Balance",
    "Salary and Benefits",
    "Job Security",
    "Promotions, Appraisal and Advancement",
    "Company Culture and Management",
    "Skill Development and Work Satisfaction"
]

# Process the DataFrame
df['Keyphrases'] = df['Review'].apply(get_keyphrases)
df[['Topic_Name', 'Topic_Relevance_Score']] = df['Keyphrases'].apply(
    lambda x: pd.Series(map_keyphrases_to_topic_name_with_score(x, topic_names))
)
df['Sentiment_Score'] = df['Review'].apply(sentiment_anlysis)
df['Sentiment_Label'] = df['Sentiment_Score'].apply(sentiment_label)
df['Keyphrases'] = df['Keyphrases'].str.split(',')

# Explode the lists into multiple rows
df_exploded = df.explode('Keyphrases', ignore_index=True)

# Merge the doc_id column from the original sheet
df_exploded = df_exploded.merge(df[['Document ID']], left_index=True, right_index=True)

# Load the existing sheet "All Reviews_Sentiment"
with pd.ExcelWriter(input_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    df_exploded.to_excel(writer, sheet_name='All Reviews_Sentiment', index=False)

print(f"Updated file {input_file} with new data in 'All Reviews_Sentiment'.")
