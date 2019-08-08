import json
import re
import operator
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams
import string

from api_keys import consumer_key, consumer_secret, access_token, access_token_secret


























#
#
# with open('JSONloc/NASATest.json', 'r') as f:
#     line = f.readline()  # read only the first tweet/line
#     tweet = json.loads(line)  # load it as Python dict
#     print(json.dumps(tweet, indent=4))  # pretty-print

# # Below lies "dumb" tokening--this doesn't know about emojis, hashtags, etc.....
# import nltk
# from nltk.tokenize import word_tokenize
#
# # nltk.download('punkt')
# tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
# print(word_tokenize(tweet))



# These strings help the program divide up the tokens in a smarter way
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


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


# tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
# print(preprocess(tweet))

# # to process all of the tweets....
#
# with open('mytweets.json', 'r') as f:
#     for line in f:
#         tweet = json.loads(line)
#         tokens = preprocess(tweet['text'])
#         do_something_else(tokens)

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']



# fname = './JSONloc/NASATest.json'
# Query Used:
# @HUDgov OR "public housing" OR "housing authority"


filename = input('Enter the file name with alphanum. and underscore (w/o extension): ')
query = input('Enter Search Query: (Options: #, @ (mention), from: (from acct), or text: )')

general_search(consumer_key, consumer_secret, access_token, access_token_secret, query, filename)

# fname = './JSONloc/willscraper_hud_2.json'
# fname = 'tweet_output_HUDtest1.json'
fname = filename+'.json'

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via'] + tokenize(query)

# Print out the stopwords so we can see what we're filtering out
print("Stopwords:")
print(stop)


with open(fname, 'r') as f:
    count_all = Counter()
    count_hashtags = Counter()
    count_terms_only = Counter()
    count_bigram = Counter()
    for line in f:
        tweet = json.loads(line)
        # Create a list with all the terms, no stopwords and no mentions
        terms_stop = [term for term in preprocess(tweet['full_text']) if (term not in stop and not term.startswith('@'))]
        # Create a list with all the hashtags
        terms_hash = [term for term in preprocess(tweet['full_text'])
                      if term.startswith('#')]
        # Count terms only (no hashtags, no mentions)
        terms_only = [term for term in preprocess(tweet['full_text'])
                      if term not in stop and not term.startswith(('#', '@'))]
        # Count bigrams (two-word phrases)
        terms_bigram = bigrams(terms_stop)

        # Update the counters
        count_all.update(terms_stop)
        count_hashtags.update(terms_hash)
        count_terms_only.update(terms_only)
        count_bigram.update(terms_bigram)

    # Print the first 5 most frequent words
    print("All (no stopwords):")
    print(count_all.most_common(10))
    print("Hashtags:")
    print(count_hashtags.most_common(10))
    print("Terms:")
    print(count_terms_only.most_common(10))
    print("Bigrams:")
    print(count_bigram.most_common(10))



