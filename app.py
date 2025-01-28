import pandas as pd
from rake_nltk import Rake
import re

df = pd.read_excel(r"D:\adani_poc\refactoring_nlp\adani_final_sentiment_output (1).xlsx")
docs = df['Review']
 
# docs = ["""Supervised learning is the machine learning task of learning a function that
#          maps an input to an output based on example input-output pairs. It infers a
#          function from labeled training data consisting of a set of training examples.
#          In supervised learning, each example is a pair consisting of an input object
#          (typically a vector) and a desired output value (also called the supervisory signal).
#          A supervised learning algorithm analyzes the training data and produces an inferred function,
#          which can be used for mapping new examples. An optimal scenario will allow for the
#          algorithm to correctly determine the class labels for unseen instances. This requires
#          the learning algorithm to generalize from the training data to unseen situations in a
#          'reasonable' way (see inductive bias).""",
       
#         """Keywords are defined as phrases that capture the main topics discussed in a document.
#         As they offer a brief yet precise summary of document content, they can be utilized for various applications.
#         In an information retrieval environment, they serve as an indication of document relevance for users, as the list
#         of keywords can quickly help to determine whether a given document is relevant to their interest.
#         As keywords reflect a document's main topics, they can be utilized to classify documents into groups
#         by measuring the overlap between the keywords assigned to them. Keywords are also used proactively
#         in information retrieval."""]
 
dc=docs[6]
 
def clean_phrase(phrase):
    # Remove non-alphanumeric characters
    phrase = re.sub(r"[^\w\s]", '', phrase)
 
    # Remove numerical strings
    phrase = re.sub(r"\b\d+\b", '', phrase)
 
    # Standardize case
    phrase = phrase.lower().strip()
 
    return phrase
 
def extract_keyphrases_rake(text, num_phrases=5, min_length=2, max_length=5):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    key_phrase = rake.get_ranked_phrases_with_scores()  # Get all ranked phrases
 
    phrase_df = pd.DataFrame()
    for kp in key_phrase:
        Key_Phrase = clean_phrase(kp[1])
        Score = kp[0]
        if min_length<=len(Key_Phrase.split())<=max_length:
            kp_dict = {'Reviews':text,'Key_Phrase':Key_Phrase,'Score':Score}
            phrase_df = pd.concat([phrase_df,pd.DataFrame([kp_dict])],ignore_index=True)
        else:
            pass    
    return phrase_df
 
 
key_phrase_df = pd.DataFrame()
for dc in docs:
    interim_keyphrase_df = extract_keyphrases_rake(dc)
    key_phrase_df = pd.concat([key_phrase_df,interim_keyphrase_df],ignore_index=True)
 
print(key_phrase_df)
key_phrase_df.to_excel('Rake_NLTK_KeyPhrase1.xlsx')