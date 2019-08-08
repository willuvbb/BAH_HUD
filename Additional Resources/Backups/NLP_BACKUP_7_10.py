# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import pickle

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
from sklearn import decomposition, ensemble
import xgboost
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

#For fancy plotting
import seaborn as sns

# Importing the dataset
input_data_1 = pd.read_csv('LaudanTweets.csv')
input_data_2 = pd.read_csv('LaudanTweets_2.csv')
input_data_3 = pd.read_csv('LaudanTweets_3.csv')
input_data_4 = pd.read_csv('LaudanTweets_4.csv')
input_data_5 = pd.read_csv('LaudanTweets_5.csv')
input_data_6 = pd.read_csv('LaudanTweets_6.csv')
input_data_7 = pd.read_csv('HughTweets_DCHA.csv')
dataset = pd.concat([input_data_1,input_data_2,input_data_3,input_data_4,input_data_5,input_data_6,input_data_7])

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

##### Formatting the output variables from categories like IR, Q, N, etc to numbers.
# Turn the output data into numbers using .factorize(), and create dictionaries to go
#   back and forth between numbers and categories.
tweets['primary_category_id'] = tweets['Primary'].factorize()[0]
tweets['secondary_category_id'] = tweets['Secondary'].factorize()[0]
from io import StringIO
primary_category_id_df = tweets[['Primary', 'primary_category_id']].drop_duplicates().sort_values('primary_category_id')
secondary_category_id_df = tweets[['Secondary', 'secondary_category_id']].drop_duplicates().sort_values('secondary_category_id')
primary_category_to_id = dict(primary_category_id_df.values)
secondary_category_to_id = dict(secondary_category_id_df.values)
primary_id_to_category = dict(primary_category_id_df[['primary_category_id', 'Primary']].values)
secondary_id_to_category = dict(secondary_category_id_df[['secondary_category_id', 'Secondary']].values)


##### NATURAL LANGUAGE PROCESSING #####
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(no_token_corpus, tweets['primary_category_id'], test_size = 0.20, random_state=0)


'''Creating the Bag of Words model: Count Vector Feature'''

def create_feature_vectors(n_features, X_train, X_test, make_tfidf=False):
    count_vect = CountVectorizer(max_features= n_features)
    count_vect.fit(X_train)
    # transform the training and validation data using count vectorizer object
    X_train_count = count_vect.transform(X_train)
    X_test_count = count_vect.transform(X_test)

    if make_tfidf:
        '''Creating the TF-IDF feature (Word-level)'''
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=n_features)
        tfidf_vect.fit(X_train)
        X_train_tfidf = tfidf_vect.transform(X_train)
        X_test_tfidf = tfidf_vect.transform(X_test)

        '''Creating the TF-IDF feature (n-gram-level)'''
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=n_features)
        tfidf_vect_ngram.fit(X_train)
        X_train_tfidf_ngram = tfidf_vect_ngram.transform(X_train)
        X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)

        '''Creating the TF-IDF feature (char-level)'''
        tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=n_features)
        tfidf_vect_ngram_char.fit(X_train)
        X_train_tfidf_ngram_char = tfidf_vect_ngram_char.transform(X_train)
        X_test_tfidf_ngram_char = tfidf_vect_ngram_char.transform(X_test)

        return X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, X_train_tfidf_ngram, \
               X_test_tfidf_ngram, X_train_tfidf_ngram_char, X_test_tfidf_ngram_char
    else:
        return X_train_count, X_test_count


# #Check for imbalanced classes?
fig = plt.figure(figsize=(8,6))
tweets.groupby('Primary').count().plot.bar(ylim=0)
plt.show()

'''the one-classifier way'''
''''# Fitting Naive Bayes to the Training set
classifier = MultinomialNB()
classifier.fit(X_train_count, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_count)

# Compute the accuracy score
my_accuracy_score = metrics.accuracy_score(y_pred, y_test)

# Making the Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)'''

# Create the success rate dataframe... we will continuously update this as we go and print for the user to see.
model_score_df = pd.DataFrame(
    columns=[
        'CNT Vec',
        'Word Lvl TF-IDF',
        'N-Gram Lvl TF-IDF',
        'Char Lvl TF-IDF'],
    dtype=float,
    index=['Naive Bayes Classifier',
           'Linear Classifier',
           'Support Vector Machine',
           'Bagging Model (Random Forest)',
           'Boosting Model (Xtreme Gradient)']
)


# Define the model training and validation function
def train_model(classifier, feature_vector_train, y_train, feature_vector_test, y_test):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(feature_vector_test)

    # Compute the accuracy score
    my_accuracy_score = metrics.accuracy_score(y_pred, y_test)

    # Making the Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    # Compute the Cohen Kappa Score
    kappa_score = cohen_kappa_score(y_test, y_pred)

    return my_accuracy_score, conf_mat, kappa_score

# A function that will score all the models for each of the feature vectors and return the model_score dataframe.
def compare_model_scores(model_score_df, X_train_count, y_train, X_test_count, y_test):
    '''------------Naive Bayes------------'''

    ''''Naive Bayes on Count Vectors'''
    accuracy, conf_mat, kappa_score = train_model(naive_bayes.MultinomialNB(), X_train_count, y_train, X_test_count, y_test)
    print("NB, Count Vectors: ", accuracy)
    # Update the model_score_df with specified clf_name, vector_name
    model_score_df.at['Naive Bayes Classifier', 'CNT Vec'] = accuracy

    '''Naive Bayes on World Level TF-IDF'''
    accuracy, conf_mat, kappa_score = train_model(naive_bayes.MultinomialNB(), X_train_tfidf, y_train, X_test_tfidf, y_test)
    print("NB, WordLevel TF-IDF: ", accuracy)
    model_score_df.at['Naive Bayes Classifier', 'Word Lvl TF-IDF'] = accuracy

    '''Naive Bayes on Ngram Level TF IDF Vectors'''
    accuracy, conf_mat, kappa_score = train_model(naive_bayes.MultinomialNB(), X_train_tfidf_ngram, y_train, X_test_tfidf_ngram, y_test)
    print("NB, N-Gram Vectors: ", accuracy)
    model_score_df.at['Naive Bayes Classifier', 'N-Gram Lvl TF-IDF'] = accuracy

    '''Naive Bayes on Character Level TF IDF Vectors'''
    accuracy, conf_mat, kappa_score = train_model(naive_bayes.MultinomialNB(), X_train_tfidf_ngram_char, y_train, X_test_tfidf_ngram_char, y_test)
    print("NB, CharLevel Vectors: ", accuracy)
    model_score_df.at['Naive Bayes Classifier', 'Char Lvl TF-IDF'] = accuracy


    '''------------Linear Classifier------------'''

    '''Linear Classifier (LogReg) on Count Vectors'''
    accuracy, conf_mat, kappa_score = train_model(linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'), X_train_count, y_train, X_test_count, y_test)
    print("LR, Count Vectors: ", accuracy)
    # Update the model_score_df with specified clf_name, vector_name
    model_score_df.at['Linear Classifier', 'CNT Vec'] = accuracy

    '''Linear Classifier (LogReg) on World Level TF-IDF'''
    accuracy, conf_mat, kappa_score = train_model(linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'), X_train_tfidf, y_train, X_test_tfidf, y_test)
    print("LR, WordLevel TF-IDF: ", accuracy)
    model_score_df.at['Linear Classifier', 'Word Lvl TF-IDF'] = accuracy

    '''Linear Classifier (LogReg) on Ngram Level TF IDF Vectors'''
    accuracy, conf_mat, kappa_score = train_model(linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'), X_train_tfidf_ngram, y_train, X_test_tfidf_ngram, y_test)
    print("LR, N-Gram Vectors: ", accuracy)
    model_score_df.at['Linear Classifier', 'N-Gram Lvl TF-IDF'] = accuracy

    '''Linear Classifier (LogReg) on Character Level TF IDF Vectors'''
    accuracy, conf_mat, kappa_score = train_model(linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'), X_train_tfidf_ngram_char, y_train, X_test_tfidf_ngram_char, y_test)
    print("LR, CharLevel Vectors: ", accuracy)
    model_score_df.at['Linear Classifier', 'Char Lvl TF-IDF'] = accuracy


    '''------------Support Vector Machine------------'''

    '''Support Vector Machine on Count Vectors'''
    accuracy, conf_mat, kappa_score = train_model(svm.LinearSVC(), X_train_count, y_train, X_test_count, y_test)
    print("SVM, Count Vectors: ", accuracy)
    # Update the model_score_df with specified clf_name, vector_name
    model_score_df.at['Support Vector Machine', 'CNT Vec'] = accuracy

    '''Support Vector Machine on World Level TF-IDF'''
    accuracy, conf_mat, kappa_score = train_model(svm.LinearSVC(), X_train_tfidf, y_train, X_test_tfidf, y_test)
    print("SVM, WordLevel TF-IDF: ", accuracy)
    model_score_df.at['Support Vector Machine', 'Word Lvl TF-IDF'] = accuracy

    '''Support Vector Machine on Ngram Level TF IDF Vectors'''
    accuracy, conf_mat, kappa_score = train_model(svm.LinearSVC(), X_train_tfidf_ngram, y_train, X_test_tfidf_ngram, y_test)
    print("SVM, N-Gram Vectors: ", accuracy)
    model_score_df.at['Support Vector Machine', 'N-Gram Lvl TF-IDF'] = accuracy

    '''Support Vector Machine on Character Level TF IDF Vectors'''
    accuracy, conf_mat, kappa_score = train_model(svm.LinearSVC(), X_train_tfidf_ngram_char, y_train, X_test_tfidf_ngram_char, y_test)
    print("SVM, CharLevel Vectors: ", accuracy)
    model_score_df.at['Support Vector Machine', 'Char Lvl TF-IDF'] = accuracy



    '''------------Bagging Model (Random Forest)------------'''

    '''Bagging Model (Random Forest) on Count Vectors'''
    accuracy, conf_mat, kappa_score = train_model(ensemble.RandomForestClassifier(n_estimators=20), X_train_count, y_train, X_test_count, y_test)
    print("RF, Count Vectors: ", accuracy)
    # Update the model_score_df with specified clf_name, vector_name
    model_score_df.at['Bagging Model (Random Forest)', 'CNT Vec'] = accuracy

    '''Bagging Model (Random Forest) on World Level TF-IDF'''
    accuracy, conf_mat, kappa_score = train_model(ensemble.RandomForestClassifier(n_estimators=20), X_train_tfidf, y_train, X_test_tfidf, y_test)
    print("RF, WordLevel TF-IDF: ", accuracy)
    model_score_df.at['Bagging Model (Random Forest)', 'Word Lvl TF-IDF'] = accuracy

    '''Bagging Model (Random Forest) on Ngram Level TF IDF Vectors'''
    accuracy, conf_mat, kappa_score = train_model(ensemble.RandomForestClassifier(n_estimators=20), X_train_tfidf_ngram, y_train, X_test_tfidf_ngram, y_test)
    print("RF, N-Gram Vectors: ", accuracy)
    model_score_df.at['Bagging Model (Random Forest)', 'N-Gram Lvl TF-IDF'] = accuracy

    '''Bagging Model (Random Forest) on Character Level TF IDF Vectors'''
    accuracy, conf_mat, kappa_score = train_model(ensemble.RandomForestClassifier(n_estimators=20), X_train_tfidf_ngram_char, y_train, X_test_tfidf_ngram_char, y_test)
    print("RF, CharLevel Vectors: ", accuracy)
    model_score_df.at['Bagging Model (Random Forest)', 'Char Lvl TF-IDF'] = accuracy


    do_xgboost = False
    if do_xgboost:
        '''------------Boosting Model (Xtreme Gradient)------------'''

        '''Boosting Model (Xtreme Gradient) on Count Vectors'''
        accuracy, conf_mat, kappa_score = train_model(xgboost.XGBClassifier(), X_train_count, y_train, X_test_count, y_test)
        print("XG, Count Vectors: ", accuracy)
        # Update the model_score_df with specified clf_name, vector_name
        model_score_df.at['Boosting Model (Xtreme Gradient)', 'CNT Vec'] = accuracy

        '''Boosting Model (Xtreme Gradient) on World Level TF-IDF'''
        accuracy, conf_mat, kappa_score = train_model(xgboost.XGBClassifier(), X_train_tfidf, y_train, X_test_tfidf, y_test)
        print("XG, WordLevel TF-IDF: ", accuracy)
        model_score_df.at['Boosting Model (Xtreme Gradient)', 'Word Lvl TF-IDF'] = accuracy

        '''Boosting Model (Xtreme Gradient) on Ngram Level TF IDF Vectors'''
        accuracy, conf_mat, kappa_score = train_model(xgboost.XGBClassifier(), X_train_tfidf_ngram, y_train, X_test_tfidf_ngram, y_test)
        print("XG, N-Gram Vectors: ", accuracy)
        model_score_df.at['Boosting Model (Xtreme Gradient)', 'N-Gram Lvl TF-IDF'] = accuracy

        '''Boosting Model (Xtreme Gradient) on Character Level TF IDF Vectors'''
        accuracy, conf_mat, kappa_score = train_model(xgboost.XGBClassifier(), X_train_tfidf_ngram_char, y_train, X_test_tfidf_ngram_char, y_test)
        print("XG, CharLevel Vectors: ", accuracy)
        model_score_df.at['Boosting Model (Xtreme Gradient)', 'Char Lvl TF-IDF'] = accuracy

    return model_score_df



'''START RUNNING STUFF'''

# For testing multiple numbers of features in the count_vect and tfidf
# features_multiplier_arr = range(1, 20)
# For running w/ just one # of features
features_multiplier_arr = [4]
accuracy_count_NB_arr = []
n_features_arr = []
kappa_score_count_NB_arr = []
for i in features_multiplier_arr:
    # print(i)
    n_features = len(X_train)*i
    do_create_features = True
    make_tfidf = True
    if do_create_features:
        if make_tfidf:
            X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, X_train_tfidf_ngram, \
            X_test_tfidf_ngram, X_train_tfidf_ngram_char, X_test_tfidf_ngram_char = create_feature_vectors(n_features, X_train, X_test, make_tfidf)
        else:
            X_train_count, X_test_count = create_feature_vectors(n_features, X_train, X_test, make_tfidf)


    do_compare_models = True
    if do_compare_models:
        scored_df = compare_model_scores(model_score_df, X_train_count, y_train, X_test_count, y_test)
        print(scored_df.to_string())

    accuracy_count_NB, conf_mat_count_NB, kappa_score_count_NB = train_model(naive_bayes.MultinomialNB(), X_train_count, y_train, X_test_count, y_test)

    n_features_arr.append(n_features)
    accuracy_count_NB_arr.append(accuracy_count_NB)
    kappa_score_count_NB_arr.append(kappa_score_count_NB)


print('n-features', n_features_arr)
print('accuracy',accuracy_count_NB_arr)
print('kappa',kappa_score_count_NB_arr)

with open('TweetDataset.pkl', 'wb') as f:
    pickle.dump(dataset,f)


#
# Displaying the confusion matrix in a ~fancy~ way
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat_count_NB, annot=True, fmt='d',
            xticklabels=primary_category_id_df.Primary.values, yticklabels=primary_category_id_df.Primary.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()