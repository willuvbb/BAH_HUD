# Import the necessary methods from tweepy library
import json
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import yaml
import pickle
import time
from os import listdir
from os.path import isfile, join
import os
import re
from tweet_config import get_app_config
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import statistics
from uszipcode import SearchEngine
from uszipcode import Zipcode
import math
from geolocation_functions import location_2_coordinates
from geolocation_functions import coordinates_2_zip
from geolocation_functions import validate_tweet_location
from geolocation_functions import get_city_state
from geolocation_functions import location_2_everything
import geopy
from geopy import geocoders
from geopy.geocoders import Nominatim
import filter_tweets as ft

# call config file
app_config = get_app_config()

class TweetExtractor(object):
    def __init__ (self, new_run=True):

        #Variables that contains the user credentials to access Twitter API 
        access_token = app_config["twitter_dev_credentials"]["access_token"]
        access_token_secret = app_config["twitter_dev_credentials"]["access_token_secret"]
        consumer_key = app_config["twitter_dev_credentials"]["consumer_key"]
        consumer_secret = app_config["twitter_dev_credentials"]["consumer_secret"]

        # call pickle
        self.pickle_path = app_config["app"]["pickle_path"]
        if new_run:
            self.tweets=[]
        else:
            print("Loading in Current Pickle")
            with open(self.pickle_path, 'rb') as f:
                self.tweets = pickle.load(f)

        #set parameters for 
        self.time_between_calls = app_config["app"]["time_between_calls"]
        self.num_calls = app_config["app"]["num_calls"]

        #This handles Twitter authetification and the connection to Twitter Search API
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth, wait_on_rate_limit=True)

        self.max_tweets = app_config["app"]["max_tweets"]
        self.search_terms = None
        # search_terms_path = app_config["app"]["search_terms_path"]

        # put geocode location into query
        latitude = str(app_config["location"]["latitude"])
        longitude = str(app_config["location"]["longitude"])
        radius = app_config["location"]["radius"]
        self.geo_string= latitude+","+longitude+","+radius
        
        # set up regular ingest job
        self.regular_job = app_config["app"]["regular_ingest"]
        self.call_limit = app_config["app"]["call_limit"]

    def pull_tweets(self, regular_job=True, return_tweets=True):
        get_json = lambda a: a._json

        if self.search_terms is None:
            #raise error
            raise ValueError("You currently have no search terms. You must set them by calling the method get_search_terms()")

        else:
            # if job is regular
            j = 0
            if regular_job == True:
                for i in range(self.num_calls):
                    for query in self.search_terms['terms']:
                        if j < self.call_limit:
                            searched_tweets = [get_json(status) for status in tweepy.Cursor(self.api.search, q=query, lang="en", tweet_mode=app_config["app"]["tweet_mode"], geocode=self.geo_string).items(self.max_tweets)]
                            print(searched_tweets[0])
                            self.tweets.extend(searched_tweets)
                            # dump back into pickle file
                            if not return_tweets:
                                print("Total tweets stored in current pickle: "+str(len(self.tweets)))
                                self.dump_tweets_to_pickle(self.pickle_path)
                            else:
                                print("Total tweets gathered thusfar: "+str(len(self.tweets)))
                            print("*"*30+" Iteration "+str(i+1)+" of the term \""+ query+"\" is complete "+"*"*30)
                            j=j+1
                            time.sleep(self.time_between_calls)

            # if job is not regular
            else: 
                for query in self.search_terms['terms']:
                    searched_tweets = [get_json(status) for status in tweepy.Cursor(self.api.search, q=query, lang="en", geocode=self.geo_string).items(self.max_tweets)]
                    self.tweets.extend(searched_tweets)
                    # dump back into pickle file
                    if not return_tweets:
                        print("Total tweets stored in current pickle: "+str(len(self.tweets)))
                        self.dump_tweets_to_pickle(self.pickle_path)
                    else:
                        print("Total tweets gathered thusfar: "+str(len(self.tweets)))                  

    def get_search_terms(self, path=app_config["app"]["search_terms_path"]):
        # load in search terms
        with open(path) as f:
            self.search_terms = yaml.load(f)

    def dump_tweets_to_pickle(self, path=app_config["app"]["pickle_path"]):
        with open(path, 'wb') as f:
            pickle.dump(self.tweets, f)

    def run(self, test_run=False):
        """ This Method is how we would like the code to run when it is in production """
        mypath = app_config["app"]["search_terms_folder"]
        rest_time = app_config["app"]["waiting_time"]
        search_term_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and re.search("[.]*[\.]yml$", f)]

        if test_run:
            search_term_files = [search_term_files[0]]

        i=0
        for _file in search_term_files:
            self.get_search_terms(path=mypath+"/"+_file)
            if test_run:
                self.search_terms["terms"] = self.search_terms["terms"][0:2]
                print("***THIS IS A TEST RUN ONLY***")
            print("Pulling down tweets with terms from "+_file)
            self.pull_tweets()
            i=i+1
            if i == len(search_term_files):
                print("This has concluded the gathering of tweets. We have gathered "+str(len(self.tweets))+ " tweets.")
            
            else:
                print("Done Pulling Tweets. Program must sleep for "+str(round(rest_time/60))+ " minutes before resuming.")
                time.sleep(rest_time-600)
                print("Program will start gathering more tweets in 10 minutes.")
                time.sleep(rest_time-300)
                print("Program will start gathering more tweets in 5 minutes.")
                time.sleep(295)
                print("5")
                time.sleep(1)
                print("4")
                time.sleep(1)
                print("3")
                time.sleep(1)
                print("2")
                time.sleep(1)
                print("1")
                time.sleep(1)

if __name__ == '__main__':
    print("creating tweet extractor object")
    extractor = TweetExtractor(new_run=True)
    print("running extractor object")
    extractor.run()
    extractor.dump_tweets_to_pickle()
    print("creating tweet ingestion object")
    ingestor = ft.TweetIngestionEngine(tweets=extractor.tweets)
    print("running tweet ingestion")
    ingestor.run()
    # merge_data()

