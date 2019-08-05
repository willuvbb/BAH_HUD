# # Scraper_Master.py
#
# Author: Will Cunningham
# Last Modified by Will Cunningham on 6/26/19

# # Import things/packages
import json
import re
import operator
from collections import Counter
from collections import defaultdict
from nltk.corpus import stopwords
from nltk import bigrams
import string
import csv
import tweepy
import sys
import re
import os
from os import listdir

from api_keys import consumer_key, consumer_secret, access_token, access_token_secret

# # Variables used for the whole program
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


# # Define functions

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

# "Search" function which scrapes twitter
def general_search(consumer_key, consumer_secret, access_token, access_token_secret, query, filename,num_tweets):
    # create authentication for accessing Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # initialize Tweepy API
    api = tweepy.API(auth)

    # get the name of the spreadsheet we will write to
    #     fname = '_'.join(re.findall(r"@(\w+)", handle_phrase))
    fname = filename

    # open the spreadsheet we will write to
    with open("./CSVloc/" + fname + ".csv", 'w') as file:

        w = csv.writer(file)

        # write header row to spreadsheet
        w.writerow(['timestamp', 'tweet_text', 'username', 'all_hashtags', 'followers_count'])

        # for each tweet matching our hashtags, write relevant info to the spreadsheet
        #         for tweet in tweepy.Cursor(api.search, q=handle_phrase+' -filter:retweets', \

        for tweet in tweepy.Cursor(api.search, q=query + ' -filter:retweets', lang="en", tweet_mode='extended').items(
                num_tweets):
            w.writerow([tweet.created_at, tweet.full_text.replace('\n', ' ').encode('utf-8'),
                        tweet.user.screen_name.encode('utf-8'),
                        [e['text'] for e in tweet._json['entities']['hashtags']], tweet.user.followers_count])

        print('The file ' + filename + '.csv has been created in your working directory.')

    with open("./JSONloc/" + fname + ".json", 'w') as f:
        for tweet in tweepy.Cursor(api.search, q=query + ' -filter:retweets', lang="en", tweet_mode='extended').items(
                num_tweets):
            #             print(type(tweet))
            f.write(json.dumps(tweet._json) + "\n")
        print('The file ' + filename + '.json has been created in your working directory.')

# "Analysis" function which does analysis
def old_analysis(filename, more_stopwords):
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via', '’','...','…'] + tokenize(more_stopwords) + \
           tokenize(more_stopwords.lower())

    fname = "./JSONloc/"+ filename + '.json'

    search_word = input("enter your search term for co-occurrences")  # pass a term as a command-line argument

    # Print out the stopwords so we can see what we're filtering out
    print("Stopwords:")
    print(stop)

    with open(fname, 'r') as f:
        count_all = Counter()
        count_hashtags = Counter()
        count_terms_only = Counter()
        count_bigram = Counter()
        count_search = Counter()
        com = defaultdict(lambda: defaultdict(int))

        for line in f:
            tweet = json.loads(line)
            # Create a list with all the terms, no stopwords and no mentions
            terms_stop = [term for term in preprocess(tweet['full_text']) if
                          (term not in stop and not term.startswith('@'))]
            # Create a list with all the hashtags
            terms_hash = [term for term in preprocess(tweet['full_text'])
                          if term.startswith('#')]
            # Count terms only (no hashtags, no mentions)
            terms_only = [term for term in preprocess(tweet['full_text'])
                          if term not in stop and not term.startswith(('#', '@'))]
            # Count bigrams (two-word phrases)
            terms_bigram = bigrams(terms_stop)

            if search_word in terms_only:
                count_search.update(terms_only)

            # Update the counters
            count_all.update(terms_stop)
            count_hashtags.update(terms_hash)
            count_terms_only.update(terms_only)
            count_bigram.update(terms_bigram)



            # Build co-occurrence matrix
            for i in range(len(terms_only) - 1):
                for j in range(i + 1, len(terms_only)):
                    w1, w2 = sorted([terms_only[i], terms_only[j]])
                    if w1 != w2:
                        com[w1][w2] += 1

        # Print the first 5 most frequent words
        print("All (Hashtags included, no stopwords, no mentions):")
        print(count_all.most_common(10))
        print("Only Hashtags:")
        print(count_hashtags.most_common(10))
        print("Terms (no hashtags, no stopwords, no mentions):")
        print(count_terms_only.most_common(10))
        print("Bigrams:")
        print(count_bigram.most_common(10))
        print("Co-occurrence for %s:" % search_word)
        print(count_search.most_common(20))



        com_max = []
        # For each term, look for the most common co-occurrent terms
        for t1 in com:
            t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
            for t2, t2_count in t1_max_terms:
                com_max.append(((t1, t2), t2_count))
        # Get the most frequent co-occurrences
        terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
        print(terms_max[:5])

def run():
    # Make a loop
    keep_running = True
    while keep_running:
        choice = input("Choose functionality. Enter ""s"" to search, ""a"" to analyze, or ""q"" to quit.")
        if choice == "q":
            keep_running = False
        elif choice == "s":
            filename = input('Enter the file name with alphanum. and underscore (w/o extension): ')
            query = input('Enter Search Query: (Options: #, @ (mention), from: (from acct), or text: )')
            num_tweets = int(input("Enter the number of tweets you wish to collect. (Max 15k): "))
            print(type(num_tweets))
            general_search(consumer_key, consumer_secret, access_token, access_token_secret, query, filename,num_tweets)
            do_analysis = input("Analyze this data? (y/n)")
            if do_analysis == "y":
                old_analysis(filename, query)

        elif choice == "a":
            print("Analysis mode. See below the list of all files available for analysis.")
            available_files = os.listdir("JSONloc")
            print(available_files)
            filename = input("Enter the filename (w/o extension) of the file you wish to analyze.")
            other_stopwords = input("If you wish to add any other stop words (which will not be analyzed), "
                                    "enter them here separated by spaces.")
            old_analysis(filename, other_stopwords)

if __name__ == '__main__':
    print("Welcome to Will's Twitter Scraper. Version 1.0.")
    run()