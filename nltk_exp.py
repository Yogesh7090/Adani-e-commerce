import nltk

# nltk.download('vader_lexicon')


# from nltk.tokenize import word_tokenize

# text = "Hello, today we will learn about Python Sentiment Analysis with NLTK."
# tokens = word_tokenize(text)
# print(tokens)

# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# text = "Hello, today we will learn about Python Sentiment Analysis with NLTK."
# tokens = word_tokenize(text)
# print("Including stop words: ", tokens)

# stop_words = set(stopwords.words('english'))
# filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
# print("Excluding stop words: ", filtered_tokens)

# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer

# text = "Python programming is becoming very popular."
# tokens = word_tokenize(text)

# stemmer = PorterStemmer()
# stemmed_tokens = [stemmer.stem(token) for token in tokens]
# print(stemmed_tokens)

# from nltk.tokenize import word_tokenize
# from nltk.stem.wordnet import WordNetLemmatizer

# text = "Python programming is becoming very popular."
# tokens = word_tokenize(text)

# lemmatizer = WordNetLemmatizer()
# lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
# print(lemmatized_tokens)

# lemmatized_tokens = [lemmatizer.lemmatize(token, "v") for token in tokens]

# print(f'{lemmatized_tokens=}')


# from nltk.sentiment import SentimentIntensityAnalyzer

# analyzer = SentimentIntensityAnalyzer()

# text = "I love this product! It's amazing."
# scores = analyzer.polarity_scores(text)
# print(f'{scores=}')

from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

review1 = "I love this product! It's amazing."
review2 = "This product is terrible. I hate it."

review1_score = analyzer.polarity_scores(review1)
print("Score for Review #1: {}".format(review1_score))
review2_score = analyzer.polarity_scores(review2)
print("Score for Review #2: {}".format(review2_score))

if review1_score['compound'] > review2_score['compound']:
    print("The review that has a more positive sentiment is Review #1: \"{}\"".format(review1))
else:
    print("The review that has a more positive sentiment is Review #2:\"{}\"".format(review2))