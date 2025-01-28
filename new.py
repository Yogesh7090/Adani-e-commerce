import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Initialize spaCy model and add the textblob pipeline
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Step 2: Load the input Excel file
df = pd.read_excel('adani_final_sentiment_output.xlsx', sheet_name='Sheet1')

# Step 3: Rename 'Unnamed: 0' to 'Doc_ID' if it exists
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'Doc_ID'}, inplace=True)

# Step 4: Define helper functions
def get_keyphrases(input_text):
    """Extract keyphrases from the input text."""
    keywords = []
    spacy_doc = nlp(input_text)
    for chunk in spacy_doc.noun_chunks:
        if chunk.text.lower() not in nlp.Defaults.stop_words:
            keywords.append(chunk.text)
    return ",".join(keywords)

def sentiment_analysis(input_text):
    """Analyze the sentiment score of the input text."""
    doc = nlp(input_text)
    sentiment = doc._.blob.polarity
    return round(sentiment, 2)

def sentiment_label(sentiment_score):
    """Label sentiment based on the sentiment score."""
    if sentiment_score > 0.2:
        return 'Positive'
    elif sentiment_score < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def map_keyphrases_to_topic_name_with_score(keyphrases, topic_names):
    """Map keyphrases to the most relevant topic name using TF-IDF and cosine similarity."""
    tfidf_vectorizer = TfidfVectorizer()
    all_text = [keyphrases] + topic_names
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_match_idx = similarity_scores.argmax()
    return topic_names[best_match_idx], similarity_scores[best_match_idx]

# Step 5: Define topic names
topic_names = [
    "Work-Life Balance",
    "Salary and Benefits",
    "Job Security",
    "Promotions, Appraisal and Advancement",
    "Company Culture and Management",
    "Skill Development and Work Satisfaction"
]

# Step 6: Process the DataFrame
df['Keyphrases'] = df['Review'].apply(get_keyphrases)  # Extract keyphrases
df[['Topic_Name', 'Topic_Relevance_Score']] = df['Keyphrases'].apply(
    lambda x: pd.Series(map_keyphrases_to_topic_name_with_score(x, topic_names))
)  # Map topics
df['Sentiment_Score'] = df['Review'].apply(sentiment_analysis)  # Get sentiment score
df['Sentiment_Label'] = df['Sentiment_Score'].apply(sentiment_label)  # Label sentiment
df['Keyphrases'] = df['Keyphrases'].str.split(',')  # Split keyphrases into lists

# Step 7: Explode keyphrases into multiple rows
df_exploded = df.explode('Keyphrases', ignore_index=True)

# Step 8: Assign a unique and ordered `Doc_ID` for each unique combination of Review, Keyphrases, and Source
df_exploded['Doc_ID'] = (
    df_exploded.groupby(['Source', 'Review', 'Keyphrases']).ngroup() + 1
)

# Sort the DataFrame by Doc_ID to maintain order
df_exploded = df_exploded.sort_values(by='Doc_ID').reset_index(drop=True)

# Step 9: Save the processed data to an Excel file
df_exploded.to_excel("unique_doc_id_for_same_reviews_with_source_6.xlsx", index=False)
print('saved')