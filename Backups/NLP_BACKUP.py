# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

# Cleaning the tweets
import re
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Import things for categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer

# Import models
from sklearn import metrics
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#For fancy plotting
import seaborn as sns

# Importing the dataset
# dataset = pd.read_csv('LaudanTweets.csv', quoting = 3)
dataset = pd.read_csv('LaudanTweets.csv')

##### PREPROCESSING #####
# Take only the columns that we want
dataset = dataset[['tweet_text','Primary','Secondary']]
# Take out the blank rows
dataset = dataset.dropna(axis=0,how='all')
# Reset the indexes
dataset = dataset.reset_index(drop=True)

'''Predefined strings and functions from MarcoBonzanini.com's tutorial. This will will split the tweets up into 
tokens in a smarter way than the standard functions will...'''
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def make_custom_stopwords():
    punctuation = list(string.punctuation)
    custom_stopwords = stopwords.words('english') + punctuation + ['rt', 'via', '’','...','…']
    return custom_stopwords

# Copy the original dataset into a DataFrame called 'tweets' just in case
tweets = dataset.copy()

# Make lists which will contain all of our words
no_token_corpus = []
token_corpus = []
# Create stopwords list
my_stopwords = make_custom_stopwords()

# Loop through all tweets and process them
ds_length = len(dataset)
for i in range(0, ds_length):
    # Take out the ___b'___ and the ___'___ at the beginning and the end of the tweet, respectively
    tweets['tweet_text'][i] = tweets['tweet_text'][i][2:-1]
    # Preprocess the tweet and save it in tweets DataFrame. "Preprocess" = "smart tokenize" using the MarcoB functions
    tweets['tweet_text'][i] = preprocess(tweets['tweet_text'][i])

    # Stem the tweets?
    ps = PorterStemmer()
    tweets['tweet_text'][i] = [ps.stem(word) for word in tweets['tweet_text'][i]]

    # Take out stopwords
    tweets['tweet_text'][i] = [word for word in tweets['tweet_text'][i] if not word in my_stopwords]

    # Join the tweet together to be added to the non-tokenized corpus
    tweet_no_token = ' '.join(tweets['tweet_text'][i])

    # Add the tweet to the "corpus", either the tokenized one (each tweet is split up) or the non-tokenized one
    #   (each tweet is a string)
    no_token_corpus.append(tweet_no_token)
    token_corpus.append(tweets['tweet_text'][i])


##### NATURAL LANGUAGE PROCESSING #####
'''Creating the Bag of Words model'''
cv = CountVectorizer(max_features = 1500)
# Turn the words stored in the non-tokenized corpus into an array corresponding to
    # the # of occurrences of each word in each tweet

''' BAD BAD BAD, THIS IS REALLY BAD BELOW. I TRIED TO FIT THE COUNT-VECTOR TO THE ENTIRE DATASET,
WHICH IS A NO-NO ... YOU ONLY FIT IT TO THE TRAINING SET!!!!!!!!!!'''
X_counts = cv.fit_transform(no_token_corpus).toarray()
output_categories = tweets[['Primary']]

'''#Check for imbalanced classes?
fig = plt.figure(figsize=(8,6))
tweets.groupby('Primary').count().plot.bar(ylim=0)
plt.show()'''

'''UPDATE: THIS CODE IS TRASH BELOW THESE TRIPLE QUOTES. DO NOT USE. 
I might not want to include the code below here. I might need to input the data differently into the multinomialNB 
object.... '''
# # Turn the classification (categorical data) into 1's and 0's using ColumnTransformer
# '''# Output: include both columns:
# output_categories = tweets[['Primary','Secondary']]
# preprocess = make_column_transformer(
#     (OneHotEncoder(), ['Primary']),
#     remainder='passthrough'
#     )'''
#
# # Output: include only primary column. Later, change the two columns into one composite column?
# output_categories = tweets[['Primary']]
#
# preprocess = make_column_transformer(
#     (OneHotEncoder(), ['Primary']),
#     remainder='passthrough'
#     )
#
# output_categories_as_num = preprocess.fit_transform(output_categories)
# y = output_categories_as_num
'''Thru here! Now start including again'''

'''An non-erroneous alternative to what I did above:'''
# Turn the column of countries into numbers, in this case, 1, 2, and 3...
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(output_categories)

'''FUTURE CHANGE: Instead of using LabelEncoder, use the method shown below, which I got online and 
is in the "consumer_complaints.ipynb" notebook.'''
# Code:
tweets['primary_category_id'] = tweets['Primary'].factorize()[0]
tweets['secondary_category_id'] = tweets['Secondary'].factorize()[0]
from io import StringIO
primary_category_id_df = tweets[['Primary', 'primary_category_id']].drop_duplicates().sort_values('primary_category_id')
secondary_category_id_df = tweets[['Secondary', 'secondary_category_id']].drop_duplicates().sort_values('secondary_category_id')
primary_category_to_id = dict(primary_category_id_df.values)
secondary_category_to_id = dict(secondary_category_id_df.values)
primary_id_to_category = dict(primary_category_id_df[['primary_category_id', 'Primary']].values)
secondary_id_to_category = dict(secondary_category_id_df[['secondary_category_id', 'Secondary']].values)
''''''

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_counts, tweets['primary_category_id'], test_size = 0.20)

'''Below here is the erroneous "second" CountVectorizer creation... I was trying to do it twice'''
# # Create a CountVectorizer feature from the dataset
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# count_vect = CountVectorizer()
# # Fit the CountVectorizer to the training data and transform the training data into a new matrix. Could have done this
# # in two lines--the first being count_vect.fit(X_train) and the second being the creation of the new variable, transformed.
# X_train_counts = count_vect.fit_transform(X_train)
# X_test_counts = count_vect.transform(X_test)
# '''tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)'''
'''Here ends the erroneous CountVectorizer...'''

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Compute the accuracy score
my_accuracy_score = metrics.accuracy_score(y_pred, y_test)

# Making the Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Displaying the confusion matrix in a ~fancy~ way
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=primary_category_id_df.Primary.values, yticklabels=primary_category_id_df.Primary.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#
# X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#
# clf = MultinomialNB().fit(X_train_tfidf, y_train)
# print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt"
#                                         " per my right under the FDCPA. I do not believe this debt is mine."])))