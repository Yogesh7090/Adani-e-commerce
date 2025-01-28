import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Spacy language model
nlp = spacy.load('en_core_web_sm')

# Load the dataset
input_file = r"D:\adani_poc\pre_processing\People Analytics_POC Plan 1 (1).xlsx"  # Replace with your file path
df = pd.read_excel(input_file, sheet_name='All Reviews')

# Handle missing values in the 'Review' column
df['Review'] = df['Review'].fillna('').astype(str)

# Function to extract keyphrases using Spacy
def get_keyphrases(input_text):
    """Extract keyphrases from text using Spacy noun chunks."""
    keywords = []
    spacy_doc = nlp(input_text)
    for chunk in spacy_doc.noun_chunks:
        if chunk.text.lower() not in nlp.Defaults.stop_words:  # Remove stop words
            keywords.append(chunk.text.strip())
    return ", ".join(keywords)

# Extract keyphrases for each review
df['keyphrases'] = df['Review'].apply(get_keyphrases)

# Define topics
topic_names = [
    "Work-Life Balance",
    "Salary and Benefits",
    "Job Security",
    "Promotions, Appraisal and Advancement",
    "Company Culture and Management",
    "Skill Development and Work Satisfaction"
]

# Function to map keyphrases to predefined topics using similarity scoring
def map_keyphrases_to_topic_name_with_score(keyphrases, topic_names):
    """Map extracted keyphrases to predefined topics using cosine similarity."""
    tfidf_vectorizer = TfidfVectorizer()
    all_text = [keyphrases] + topic_names  # First element is the keyphrases
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_match_idx = similarity_scores.argmax()
    best_topic = topic_names[best_match_idx]
    best_score = similarity_scores[best_match_idx]
    return best_topic, best_score

# Apply topic mapping and get scores for each review
df[['Topic_Name', 'Score']] = df['keyphrases'].apply(
    lambda x: pd.Series(map_keyphrases_to_topic_name_with_score(x, topic_names))
)

# Add a Document_ID column
df['Document_ID'] = range(1, len(df) + 1)

# Save the output to a CSV file
output_file = r"D:\adani_poc\pre_processing\adani_updated_version_01.csv"
output_columns = ['Review', 'keyphrases', 'Topic_Name', 'Score', 'Document_ID']
df[output_columns].to_csv(output_file, index=False)

# Print a sample of the output
print(df[output_columns].head())
