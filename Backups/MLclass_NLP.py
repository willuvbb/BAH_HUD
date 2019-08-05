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

mypath = 'ClassifiedTweets'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

new_dataset = pd.DataFrame()

for input_file in onlyfiles:
    print(input_file)
    small_df = pd.read_csv(mypath+'/'+input_file)
    new_dataset = pd.concat([new_dataset,small_df],ignore_index=True)
#
# # Importing the dataset
# input_data_1 = pd.read_csv(mypath+'/'+'LaudanTweets.csv')
# input_data_2 = pd.read_csv(mypath+'/'+'LaudanTweets_2.csv')
# input_data_3 = pd.read_csv(mypath+'/'+'LaudanTweets_3.csv')
# input_data_4 = pd.read_csv(mypath+'/'+'LaudanTweets_4.csv')
# input_data_5 = pd.read_csv(mypath+'/'+'LaudanTweets_5.csv')
# input_data_6 = pd.read_csv(mypath+'/'+'LaudanTweets_6.csv')
# input_data_7 = pd.read_csv(mypath+'/'+'HughTweets_DCHA.csv')
# input_data_8 = pd.read_csv(mypath+'/'+'LaudanTweets_7.csv')
# input_data_9 = pd.read_csv(mypath+'/'+'HughTweets_NYCHA.csv')
# input_data_10 = pd.read_csv(mypath+'/'+'HughTweets_NYCHA_2.csv')
#
# dataset = pd.concat([input_data_1,input_data_2,input_data_3,input_data_4,input_data_5,input_data_6,input_data_7,
#                      input_data_8, input_data_9, input_data_10])

##### PREPROCESSING #####
# Take only the columns that we want
dataset = new_dataset[['tweet_text','Primary','Secondary']]
# Take out the blank rows
dataset = dataset.dropna(axis=0,how='all')
# Reset the indexes
dataset = dataset.reset_index(drop=True)

# Check initial dataset size
init_shape = dataset.shape
# Turn Hugh's "I" label for irrelevant and turn it to "IR" for consistency with Laudan's formatting
dataset['Primary'] = dataset['Primary'].str.replace('IR', 'I',regex=True, case = False)
# # Handle the 'nan's in the secondary column
# dataset['Secondary'] = dataset['Secondary'].str.replace(np.nan, 'I',regex=True, case = False)
# Define keywords that we don't want to have in the dataset
keywords_to_remove = ['australia','jacqui','tasmania','hong kong']
# Change the dataset to lowercase for matching the keywords
dataset['tweet_text'] = dataset['tweet_text'].str.lower()
# Take out rows with the keywords
row_has_keyword = [any([i.find(phrase) != -1 for phrase in keywords_to_remove]) for i in dataset["tweet_text"]]
indexes_to_drop = [i for i, x in enumerate(row_has_keyword) if x == True]
dataset = dataset.drop(indexes_to_drop, axis=0)

# Combine maintenance and experience
dataset['Primary'] = dataset['Primary'].str.replace('M', 'E',regex=True, case = False)
# Combine policy decision and news
dataset['Primary'] = dataset['Primary'].str.replace('PD', 'N',regex=True, case = False)


# Take out the blank rows
dataset = dataset.dropna(axis=0,how='any')
# Reset the indexes
dataset = dataset.reset_index(drop=True)
# Check dataset size after dropping rows w/ keywords
final_shape = dataset.shape

# Save the dataset as a csv
dataset.to_csv('AllTweets.csv')

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
    # take out ? and ! from the list of stopwords (for question identification)
    custom_stopwords.remove('?')
    custom_stopwords.remove('!')
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
X_train, X_test, y_train, y_test = train_test_split(no_token_corpus, tweets['primary_category_id'], test_size = 0.30, random_state=0)


'''Creating the Bag of Words model: Count Vector Feature'''

def create_feature_vectors(n_features, X_train, X_test, make_tfidf=True):
    count_vect = CountVectorizer(max_features= n_features,token_pattern=r"(?u)\b\w\w+\b|!|\?")
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
def train_model(classifier, classifier_title, feature_vector_train, feature_vector_test, feature_title,
                y_train, y_test, model_score_dict):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(feature_vector_test)

    # Compute the accuracy score and store in dict
    model_score_dict[classifier_title][feature_title]['accuracy'] = metrics.accuracy_score(y_pred, y_test)

    # Making the Confusion Matrix and store in dict
    model_score_dict[classifier_title][feature_title]['conf_mat'] = confusion_matrix(y_test, y_pred)

    # Compute the Cohen Kappa Score and store in dict
    model_score_dict[classifier_title][feature_title]['kappa'] = cohen_kappa_score(y_test, y_pred)

    return model_score_dict

'''START RUNNING STUFF'''
n_features = len(X_train)*4

X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, X_train_tfidf_ngram, \
X_test_tfidf_ngram, X_train_tfidf_ngram_char, X_test_tfidf_ngram_char = create_feature_vectors(n_features, X_train,
                                                                                               X_test, make_tfidf=True)

classifier_list = [(naive_bayes.MultinomialNB(), 'Naive Bayes Classifier'),
                   (linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'), 'Linear Classifier'),
                   (svm.LinearSVC(), 'Support Vector Machine'),
                   (ensemble.RandomForestClassifier(n_estimators=20), 'Bagging Model (Random Forest)'),
                   (xgboost.XGBClassifier(), 'Boosting Model (Xtreme Gradient)')]

feature_list = [(X_train_count, X_test_count, 'CNT Vec'),
                (X_train_tfidf, X_test_tfidf, 'Word Lvl TF-IDF'),
                (X_train_tfidf_ngram, X_test_tfidf_ngram, 'N-Gram Lvl TF-IDF'),
                (X_train_tfidf_ngram_char, X_test_tfidf_ngram_char, 'Char Lvl TF-IDF')]

scores_dict = {
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


# Resample the data to correct for class imbalance
# Under-Sampling: Tomek Links

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

for classy in classifier_list:
    for feat in feature_list:
        scores_dict = train_model(*classy, *feat, y_train, y_test, scores_dict)
        # print(scores_dict)
        # print(" ")
        # EXAMPLE:
        # scores_dict['Naive Bayes Classifier']['CNT Vec']['conf_mat']
        # Print out the classifier and feature title
        print(classy[1], feat[2],': ')
        print('Accuracy: ', scores_dict[classy[1]][feat[2]]['accuracy'])
        print('Confusion Matrix: ')
        print(scores_dict[classy[1]][feat[2]]['conf_mat'])

print_fancy_cm = True
if print_fancy_cm:
    for classy in classifier_list:
        for feat in feature_list:
            print(classy[1], feat[2],': ')
            print('Accuracy: ', scores_dict[classy[1]][feat[2]]['accuracy'])
            # Displaying the confusion matrix in a ~fancy~ way
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(scores_dict[classy[1]][feat[2]]['conf_mat'], annot=True, fmt='d',
                        xticklabels=primary_category_id_df.Primary.values,
                        yticklabels=primary_category_id_df.Primary.values)
            plt.title('Conf Mat:' + classy[1] + ', ' + feat[2])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()

with open('TweetDataset.pkl', 'wb') as f:
    pickle.dump(dataset,f)

# Displaying the confusion matrix in a ~fancy~ way
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(scores_dict[classy[1]][feat[2]]['conf_mat'], annot=True, fmt='d',
            xticklabels=primary_category_id_df.Primary.values, yticklabels=primary_category_id_df.Primary.values)
plt.title('Conf Mat:'+classy[1]+', '+feat[2])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()