# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import pickle
import imblearn

# Cleaning the tweets
import re
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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

from os import listdir
from os.path import isfile, join

# Elasticsearch stuff
from elasticsearch import helpers, Elasticsearch
import csv

# Tweepy stuff
import tweepy
from api_keys import consumer_key, consumer_secret, access_token, access_token_secret

def preprocess(s, lowercase=True):
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
    tokens = tokens_re.findall(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def make_custom_stopwords():
    punctuation = list(string.punctuation)
    custom_stopwords = stopwords.words('english') + punctuation + ['rt', 'via', '’', '...', '…']
    # take out ? and ! from the list of stopwords (for question identification)
    custom_stopwords.remove('?')
    custom_stopwords.remove('!')
    return custom_stopwords

# class Category(object):
#     def __init__(self, cat_name):
#         self.name = cat_name
#
#         if self.name is 'Primary':
#             self.column_id_name = primary_id_columnName
#             self.id_df = primary_category_id_df
#         elif self.name is 'Secondary':
#             self.column_id_name = secondary_id_columnName
#             self.id_df = secondary_category_id_df

class Classifier(object):
    def __init__(self, cat_name,use_last_import=False):

        '''Load in all of the csv files in the "classifiedTweets" folder'''
        self.mypath = 'ClassifiedTweets'
        self.onlyfiles = [f for f in listdir(self.mypath) if isfile(join(self.mypath, f))]
        self.dataset = pd.DataFrame()

        self.tweets = None
        self.no_token_corpus = None

        self.active_cat = {'name': cat_name,
                           'column_id_name': '',
                           'id_df': ''}

        self.scores_dict = {
            'Naive Bayes Classifier': {
                'CNT Vec': {},
                'Word Lvl TF-IDF': {},
                'N-Gram Lvl TF-IDF': {},
                'Char Lvl TF-IDF': {}
            },
            'Linear Classifier': {
                'CNT Vec': {},
                'Word Lvl TF-IDF': {},
                'N-Gram Lvl TF-IDF': {},
                'Char Lvl TF-IDF': {}
            },
            'Support Vector Machine': {
                'CNT Vec': {},
                'Word Lvl TF-IDF': {},
                'N-Gram Lvl TF-IDF': {},
                'Char Lvl TF-IDF': {}
            },
            'Bagging Model (Random Forest)': {
                'CNT Vec': {},
                'Word Lvl TF-IDF': {},
                'N-Gram Lvl TF-IDF': {},
                'Char Lvl TF-IDF': {}
            },
            'Boosting Model (Xtreme Gradient)': {
                'CNT Vec': {},
                'Word Lvl TF-IDF': {},
                'N-Gram Lvl TF-IDF': {},
                'Char Lvl TF-IDF': {}
            }
        }

        self.classifier_list = None
        self.feature_list = None

    def import_tweets(self):
        for input_file in self.onlyfiles:
            # print(input_file)
            small_df = pd.read_csv(self.mypath+'/'+input_file)
            self.dataset = pd.concat([self.dataset, small_df], ignore_index=True)

    def filter_dataset(self):
        '''Preprocess the tweets'''

        self.dataset = self.dataset[['tweet_text','Primary','Secondary','timestamp','username','followers_count']]
        # Take out the blank rows
        self.dataset = self.dataset.dropna(axis=0,how='all')
        # Reset the indexes
        self.dataset = self.dataset.reset_index(drop=True)

        # Check initial dataset size
        init_shape = self.dataset.shape
        # Turn Hugh's "I" label for irrelevant and turn it to "IR" for consistency with Laudan's formatting
        self.dataset['Primary'] = self.dataset['Primary'].str.replace('IR', 'I',regex=True, case = False)
        # Combine emotional anger and frustration categories
        self.dataset['Secondary'] = self.dataset['Secondary'].str.replace('EA', 'F', regex=True, case=False)
        # Combine supportive and positive categories
        self.dataset['Secondary'] = self.dataset['Secondary'].str.replace('S', 'P', regex=True, case=False)

        # Define keywords that we don't want to have in the dataset
        keywords_to_remove = ['australia','jacqui','tasmania','hong kong', 'victorian']
        # Change the dataset to lowercase for matching the keywords
        dataset_lower = self.dataset.copy()
        dataset_lower['tweet_text'] = dataset_lower['tweet_text'].str.lower()
        # Take out rows with the keywords
        row_has_keyword = [any([i.find(phrase) != -1 for phrase in keywords_to_remove]) for i in dataset_lower["tweet_text"]]
        indexes_to_drop = [i for i, x in enumerate(row_has_keyword) if x is True]
        self.dataset = self.dataset.drop(indexes_to_drop, axis=0)

        # Take out the blank rows
        self.dataset = self.dataset.dropna(axis=0,how='any')
        # Reset the indexes
        self.dataset = self.dataset.reset_index(drop=True)
        # Check dataset size after dropping rows w/ keywords
        final_shape = self.dataset.shape

        # Copy the original dataset into a DataFrame called 'tweets' to be used for NLP. Doing it here and not in the
        # preprocess_tweets() function to ensure that tweets is saved now, and not after messing with self.dataset
        # for display in the table in the webapp
        self.tweets = self.dataset.copy()

    def process_dataset_for_export(self):
        # Dictionaries to turn the abbreviations for categories into words
        prim_abb_to_string = {
            "N": "News",
            "C": "Comment",
            "PD": "Policy Decision",
            "I": "Irrelevant",
            "E": "Experience",
            "Q": "Question",
            "M": "Maintenance"
        }

        sec_abb_to_string = {
            "N": "Neutral",
            "F": "Frustrated",
            "P": "Positive",
            "C": "Confused"
        }

        ds_length = len(self.dataset)
        for i in range(0, ds_length):
            # Take out the ___b'___ and the ___'___ at the beginning and the end of the tweet, respectively
            self.dataset.at[i, 'tweet_text'] = self.dataset.at[i,'tweet_text'][2:-1]
            # Take out the ___b'___ and the ___'___ at the beginning and the end of the username, respectively
            self.dataset.at[i, 'username'] = self.dataset.at[i,'username'][2:-1]
            # Convert the primary category
            self.dataset.at[i, 'Primary'] = prim_abb_to_string[self.dataset.at[i, 'Primary']]
            # Convert the secondary category
            self.dataset.at[i, 'Secondary'] = sec_abb_to_string[self.dataset.at[i, 'Secondary']]

    def get_location_data(self):
        # create authentication for accessing Twitter
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        # initialize Tweepy API
        api = tweepy.API(auth)

        # add new column to dataframe
        self.dataset['Location'] = 'N/A'

        # Make a dict of users and locations
        with open('GetUserLocDict.pkl', 'rb') as load_dict:
            get_user_loc_dict = pickle.load(load_dict)

        # Loop through dataset and get location for each tweet (via "user")
        ds_length = len(self.dataset)
        for i in range(0, ds_length):
            print(i)
            temp_username = self.dataset.at[i, 'username']
            if temp_username not in get_user_loc_dict:
                try:
                    temp_user = api.get_user(temp_username)
                except tweepy.error.TweepError:
                    print("Tweepy Error")
                    get_user_loc_dict[temp_username] = ''
                except:
                    print("Something else went wrong")
                else:
                    temp_loc = temp_user._json['location']
                    # if temp_loc is not '':
                    get_user_loc_dict[temp_username] = temp_loc

            self.dataset.at[i, 'Location'] = get_user_loc_dict[self.dataset.at[i, 'username']]

        with open('GetUserLocDict.pkl', 'wb') as save_dict:
            pickle.dump(get_user_loc_dict, save_dict)

    def preprocess_tweets(self):
        # Make lists which will contain all of our words
        self.no_token_corpus = []
        token_corpus = []
        # Create stopwords list
        my_stopwords = make_custom_stopwords()

        # Loop through all tweets and process them
        ds_length = len(self.dataset)
        for i in range(0, ds_length):
            # Take out the ___b'___ and the ___'___ at the beginning and the end of the tweet, respectively
            self.tweets.at[i, 'tweet_text'] = self.tweets.at[i,'tweet_text'][2:-1]
            # Preprocess the tweet and save it in tweets DataFrame. "Preprocess" = "smart tokenize" using the MarcoB functions
            self.tweets.at[i,'tweet_text'] = preprocess(self.tweets.at[i,'tweet_text'])

            # Stem the tweets?
            ps = PorterStemmer()
            self.tweets.at[i,'tweet_text'] = [ps.stem(word) for word in self.tweets.at[i,'tweet_text']]

            # Take out stopwords
            self.tweets.at[i,'tweet_text'] = [word for word in self.tweets.at[i,'tweet_text'] if not word in my_stopwords]

            # Join the tweet together to be added to the non-tokenized corpus
            tweet_no_token = ' '.join(self.tweets.at[i,'tweet_text'])

            # Add the tweet to the "corpus", either the tokenized one (each tweet is split up) or the non-tokenized one
            #   (each tweet is a string)
            self.no_token_corpus.append(tweet_no_token)
            token_corpus.append(self.tweets.at[i,'tweet_text'])

    def save_tweets_to_csv(self):
        # Save the dataset as a csv to use in the web app!
        self.dataset.to_csv('AllTweets.csv',index=False)

    def save_tweets_to_json(self):
        self.dataset.to_json('AllTweets.json',orient='records')

    def save_to_elasticsearch(self):
        es = Elasticsearch()

        # delete the index "twitter", ignoring if it's already deleted/doesn't exist (ignore 404 and 400)
        es.indices.delete(index='twitter', ignore=[400, 404])
        # create the index
        es.indices.create(index='twitter')

        # Save the data to the index
        with open('AllTweets.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index='twitter', doc_type='tweets')


    def category_init(self):
        ##### Formatting the output variables from categories like IR, Q, N, etc to numbers.
        # Turn the output data into numbers using .factorize(), and create dictionaries to go
        #   back and forth between numbers and categories.
        primary_id_columnName = 'primary_category_id'
        self.tweets[primary_id_columnName] = self.tweets['Primary'].factorize()[0]
        primary_category_id_df = self.tweets[['Primary', primary_id_columnName]].drop_duplicates().sort_values(primary_id_columnName)
        primary_category_to_id = dict(primary_category_id_df.values)
        primary_id_to_category = dict(primary_category_id_df[[primary_id_columnName, 'Primary']].values)

        secondary_id_columnName = 'secondary_category_id'
        self.tweets[secondary_id_columnName] = self.tweets['Secondary'].factorize()[0]
        secondary_category_id_df = self.tweets[['Secondary', secondary_id_columnName]].drop_duplicates().sort_values(secondary_id_columnName)
        secondary_category_to_id = dict(secondary_category_id_df.values)
        secondary_id_to_category = dict(secondary_category_id_df[[secondary_id_columnName, 'Secondary']].values)

        ##### NATURAL LANGUAGE PROCESSING #####
        # Set the "active" category--that which we will be classifying
        if self.active_cat['name'] is 'Primary':
            self.active_cat['column_id_name'] = primary_id_columnName
            self.active_cat['id_df'] = primary_category_id_df
        elif self.active_cat['name'] is 'Secondary':
            self.active_cat['column_id_name'] = secondary_id_columnName
            self.active_cat['id_df'] = secondary_category_id_df

    '''Creating the Bag of Words model: Count Vector Feature'''
    def create_feature_vectors(self,n_features, X_train, X_test, make_tfidf=True):
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

    # # #Check for imbalanced classes?
    # fig = plt.figure(figsize=(8,6))
    # self.tweets.groupby(active_category).count().plot.bar(ylim=0)
    # plt.show()

    #
    # # Create the success rate dataframe... we will continuously update this as we go and print for the user to see.
    # model_score_df = pd.DataFrame(
    #     columns=[
    #         'CNT Vec',
    #         'Word Lvl TF-IDF',
    #         'N-Gram Lvl TF-IDF',
    #         'Char Lvl TF-IDF'],
    #     dtype=float,
    #     index=['Naive Bayes Classifier',
    #            'Linear Classifier',
    #            'Support Vector Machine',
    #            'Bagging Model (Random Forest)',
    #            'Boosting Model (Xtreme Gradient)']
    # )


    # Define the model training and validation function

    def train_model(self,classifier, classifier_title, feature_vector_train, feature_vector_test, feature_title,
                    y_train, y_test):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(feature_vector_test)

        # Compute the accuracy score and store in dict
        self.scores_dict[classifier_title][feature_title]['accuracy'] = metrics.accuracy_score(y_pred, y_test)

        # Making the Confusion Matrix and store in dict
        self.scores_dict[classifier_title][feature_title]['conf_mat'] = confusion_matrix(y_test, y_pred)

        # Compute the Cohen Kappa Score and store in dict
        self.scores_dict[classifier_title][feature_title]['kappa'] = cohen_kappa_score(y_test, y_pred)

    '''START RUNNING STUFF'''
    def classify(self):
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(self.no_token_corpus,
                                                            self.tweets[self.active_cat['column_id_name']],
                                                            test_size=0.30, random_state=0)

        # The number of features to be used in the count vector and tfidf vector is 4 times the # of tweets
        n_features = len(X_train)*4

        # Make the feature vectors
        X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, X_train_tfidf_ngram, \
        X_test_tfidf_ngram, X_train_tfidf_ngram_char, X_test_tfidf_ngram_char = self.create_feature_vectors(
            n_features, X_train, X_test, make_tfidf=True)
        # Make lists of classifiers to be used in the dictionary which stores the scores and confusion matrices
        self.classifier_list = [(naive_bayes.MultinomialNB(), 'Naive Bayes Classifier'),
                           (linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'), 'Linear Classifier'),
                           (svm.LinearSVC(), 'Support Vector Machine'),
                           (ensemble.RandomForestClassifier(n_estimators=20), 'Bagging Model (Random Forest)'),
                           (xgboost.XGBClassifier(), 'Boosting Model (Xtreme Gradient)')]

        self.feature_list = [(X_train_count, X_test_count, 'CNT Vec'),
                        (X_train_tfidf, X_test_tfidf, 'Word Lvl TF-IDF'),
                        (X_train_tfidf_ngram, X_test_tfidf_ngram, 'N-Gram Lvl TF-IDF'),
                        (X_train_tfidf_ngram_char, X_test_tfidf_ngram_char, 'Char Lvl TF-IDF')]



        # If you want to address class imbalancing, set do_resample to True
        do_resample = False
        if do_resample:
            from imblearn.under_sampling import TomekLinks
            tomek_links = TomekLinks(return_indices=True, ratio='majority')
            X_train_count_tl, y_train_tl, id_tl = tomek_links.fit_sample(X_train_count, y_train)
            print('Tomek links: Removed indexes:', id_tl)
            scores_dict = train_model(naive_bayes.MultinomialNB(), 'Naive Bayes Classifier', X_train_count_tl, X_test_count, 'CNT Vec', y_train_tl, y_test, scores_dict)
            print('Accuracy: ', scores_dict['Naive Bayes Classifier']['CNT Vec']['accuracy'])
            print('Confusion Matrix: \n', scores_dict['Naive Bayes Classifier']['CNT Vec']['conf_mat'])

            # Over-Sampling: SMOTE
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(ratio='minority')
            X_train_count_sm, y_train_sm = smote.fit_sample(X_train_count, y_train)
            print('SMOTE: Oversampling:')
            scores_dict = train_model(naive_bayes.MultinomialNB(), 'Naive Bayes Classifier', X_train_count_sm, X_test_count, 'CNT Vec', y_train_sm, y_test, scores_dict)
            print('Accuracy: ', scores_dict['Naive Bayes Classifier']['CNT Vec']['accuracy'])
            print('Confusion Matrix: \n', scores_dict['Naive Bayes Classifier']['CNT Vec']['conf_mat'])

            # Sci-kit learn : resample
            from sklearn.utils import resample
            X_train_count_rs, y_train_rs = resample(X_train_count, y_train)
            print('Sci-kit learn resampling:')
            scores_dict = train_model(naive_bayes.MultinomialNB(), 'Naive Bayes Classifier', X_train_count_rs, X_test_count, 'CNT Vec', y_train_rs, y_test, scores_dict)
            print('Accuracy: ', scores_dict['Naive Bayes Classifier']['CNT Vec']['accuracy'])
            print('Confusion Matrix: \n', scores_dict['Naive Bayes Classifier']['CNT Vec']['conf_mat'])

            # Run the code just once, for the NB, CNT Vec
            # STANDARD RESULT; NO RESAMPLING
            print('Standard Results; No resampling.')
            scores_dict = train_model(naive_bayes.MultinomialNB(), 'Naive Bayes Classifier', X_train_count, X_test_count, 'CNT Vec', y_train, y_test, scores_dict)
            print('Accuracy: ', scores_dict['Naive Bayes Classifier']['CNT Vec']['accuracy'])
            print('Confusion Matrix: \n', scores_dict['Naive Bayes Classifier']['CNT Vec']['conf_mat'])

        # Train and score every single classifier with every single feature, and print their accuracies/conf mats
        for classy in self.classifier_list:
            for feat in self.feature_list:
                self.train_model(*classy, *feat, y_train, y_test)
                # print(scores_dict)
                # print(" ")
                # EXAMPLE:
                # scores_dict['Naive Bayes Classifier']['CNT Vec']['conf_mat']
                # Print out the classifier and feature title
                print(classy[1], feat[2],self.active_cat['name'],': ')
                # print('Confusion Matrix: ')
                # print(scores_dict[classy[1]][feat[2]]['conf_mat'])
                print('Accuracy: ', self.scores_dict[classy[1]][feat[2]]['accuracy'])

    def print_fancy_cm(self):
        for classy in self.classifier_list:
            for feat in self.feature_list:
                print(classy[1], feat[2],': ')
                print('Accuracy: ', self.scores_dict[classy[1]][feat[2]]['accuracy'])
                # Displaying the confusion matrix in a ~fancy~ way
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(self.scores_dict[classy[1]][feat[2]]['conf_mat'], annot=True, fmt='d',
                            xticklabels=self.active_cat['id_df'][self.active_cat['name']].values,
                            yticklabels=self.active_cat['id_df'][self.active_cat['name']].values)
                plt.title('Conf Mat:' + classy[1] + ', ' + feat[2])
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.show()



    def run(self):
        # call other methods!
        self.import_tweets()
        self.filter_dataset()
        self.preprocess_tweets()

        self.process_dataset_for_export()
        self.get_location_data()

        self.save_tweets_to_csv()
        self.save_tweets_to_json()
        self.category_init()
        self.classify()
        self.print_fancy_cm()

    def run_data_export_only(self):
        self.import_tweets()
        self.filter_dataset()
        self.preprocess_tweets()
        self.process_dataset_for_export()
        self.get_location_data()
        self.save_tweets_to_csv()
        self.save_tweets_to_json()
        # self.save_to_elasticsearch()
        # self.category_init()

if __name__ == '__main__':
    # print("Creating new classifier object.")
    # primary_classifier = Classifier(cat_name='Primary')
    # print("Running classifier object.")
    # primary_classifier.run()
    print("Creating new classifier object.")
    secondary_classifier = Classifier(cat_name='Secondary')
    print("Running classifier object.")
    secondary_classifier.run_data_export_only()
