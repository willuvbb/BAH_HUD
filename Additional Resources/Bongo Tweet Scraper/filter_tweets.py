import pickle
import geopy
from geopy import geocoders
from tweet_config import get_app_config
from geopy.geocoders import Nominatim
import re
import yaml
import pandas as pd
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

# call config file
app_config = get_app_config()

get_long = lambda a: a[0]
get_lat = lambda a: a[1]
rt = app_config["app"]["retweets"]
plot_it = app_config["app"]["get_plot"]
pkl = app_config["app"]["get_df_pickle"]
columns=['id','text','timestamp','user_location', "latitude", "longitude", "zipcode", "county", "city", "state", "city_state"]
def get_text(a):
    return a["full_text"]

class TweetIngestionEngine(object):
    """This object ingests tweets the get_tweets.py script loaded into the pickle file and creates a csv from them"""
    def __init__(self, tweets=None, tweet_df=pd.DataFrame(columns=columns), retweets=rt, get_plot=plot_it, get_df_pickle=pkl):
        """initializes values via lazy loading for necessary parts of the pipeline"""

        columns_set={'id','full_text','timestamp','user_location', "latitude", "longitude", "zipcode", "county", "city", "state", "city_state"}

        #get necessary paths
        pickle_path = app_config["app"]["pickle_path"]
        zip_fip_path = app_config["app"]["zip_fip_path"]
        geolocator_app_name = app_config["geolocator"]["app_name"]
        geolocator_timeout = app_config["geolocator"]["timeout"]
        do_geolocation = app_config["geolocator"]["locations_to_coordinates"]
        coords_to_zip_fip = app_config["geolocator"]["coordinates_to_zip_fip"]
        state_lookup_path = app_config["app"]["state_lookup_path"]
        if tweets is None:
            with open(pickle_path, 'rb') as f:
                self.tweets = pickle.load(f)
                print(type(self.tweets))
        else:
            self.tweets = tweets
        
        # initialize dataframe and other attributes
        self.tweet_df = tweet_df
        df_cols = {col for col in list(self.tweet_df)}
        
        for col in columns_set - df_cols:
            self.tweet_df[col] = None

        for col in df_cols - columns_set:
            self.tweet_df.drop(col, axis=1)
        
        self.analyzer = SentimentIntensityAnalyzer()
        self.get_plot = get_plot
        self.retweets = retweets
        self.get_df_pickle = get_df_pickle
        self.do_geolocation = do_geolocation
        self.coords_to_zip_fip = coords_to_zip_fip
        if self.do_geolocation:
            self.geolocator = Nominatim(user_agent=geolocator_app_name, timeout=geolocator_timeout)
        if self.coords_to_zip_fip:
            self.search_zip = SearchEngine(simple_zipcode=True) 
            self.zip_fip_lookup = pd.read_csv(zip_fip_path)
            self.state_lookup = pd.read_csv(state_lookup_path)
        
    def tweet_filter_func(self):
        """filters out repeat tweets and tweets without valid location data"""
        unique_tweets=[]
        if self.retweets:
            for tweet in self.tweets:
                if tweet not in unique_tweets:
                    unique_tweets.append(tweet)

        else:
            tweet_text = []
            for tweet in self.tweets:
                if tweet["full_text"] not in tweet_text:
                    unique_tweets.append(tweet)
                    tweet_text.append(tweet["full_text"])

        # path=app_config['app']['remove_locations_path']
        # with open(path) as f:
        #     locations = yaml.load(f)
        # filter_words=locations["invalid_locations"]

        new_tweets=[x for x in unique_tweets if x['user']['location']!="" or x["geo"] is not None]
        # for word in filter_words:
        #     new_tweets=[x for x in new_tweets if x["geo"] is not None or re.search(word, x['user']['location'].strip().lower()) is None]

        return new_tweets

    def scored_tweets(self, tweets):
        """calls Vader to provide for different measures of sentiment to a given tweet"""
        return {"compound_score": [self.analyzer.polarity_scores(x["full_text"])["compound"] for x in tweets], "positive_score": [self.analyzer.polarity_scores(x["full_text"])["pos"] for x in tweets], "negative_score": [self.analyzer.polarity_scores(x["full_text"])["neg"] for x in tweets], "neutral_score": [self.analyzer.polarity_scores(x["full_text"])["neu"] for x in tweets]}

    def tweets_to_df(self):
        """converts tweets to a dataframe/csv with options to output a sentiment histogram and a dataframe pickle"""
        tweet_json = self.tweet_filter_func()
        sentiment_scores = self.scored_tweets(tweet_json)

        i=0
        for row_data in tweet_json:
            row_id = row_data["id"]
            row_text = row_data["full_text"]
            row_timestamp = row_data["created_at"]
            row_user_location = row_data["user"]["location"]
            if row_data["coordinates"] is None:
                row_latitude = None
                row_longitude = None
                if row_data["place"] is not None:
                    box = row_data["place"]["bounding_box"]["coordinates"][0]
                    longs = [get_long(x) for x in box]
                    lats = [get_lat(x) for x in box]
                    row_latitude = statistics.mean(lats)
                    row_longitude = statistics.mean(longs)
                # else:
                #     if self.do_geolocation:
                #         new_coords = location_2_coordinates(row_user_location, self.geolocator)
                #         row_latitude = new_coords[0]
                #         row_longitude = new_coords[1]

            else:
                row_latitude = row_data["coordinates"]["coordinates"][1]
                row_longitude = row_data["coordinates"]["coordinates"][0]
            row = pd.Series({'id': row_id, 'full_text': row_text, 'timestamp': row_timestamp, 'user_location': row_user_location,
            'latitude': row_latitude, 'longitude': row_longitude})
            
            self.tweet_df.loc[i] = row
            i = i+1

        self.tweet_df["compound_sentiment"]=sentiment_scores["compound_score"]
        self.tweet_df["positive_sentiment"]=sentiment_scores["positive_score"]
        self.tweet_df["negative_sentiment"]=sentiment_scores["negative_score"]
        self.tweet_df["neutral_sentiment"]=sentiment_scores["neutral_score"]

        print("Starting Geolocation Process")
        if self.do_geolocation:
            # code for geoloaction
            self.geolocating()

        if self.coords_to_zip_fip:
            # code for coordinates to zip and county
            self.coordinates_to_zip_county()
            self.location_from_profile()

        self.twitter_data_2_csv()

        if self.get_plot:
            self.plot_sentiment(sentiment_scores["compound_score"])

        if self.get_df_pickle:
            self.tweet_df_2_pickle()

        print("tweets with 0 sentiment: "+str(len([x for x in sentiment_scores["compound_score"] if x == 0])))
        print("\ntweets that now have an associated location: "+str(len([x for x in list(self.tweet_df["latitude"]) if x is not None])))

    def plot_sentiment(self, sentiment_scores):
        """plots sentiment histogram"""
        # An "interface" to matplotlib.axes.Axes.hist() method
        print(len(sentiment_scores))
        percentile_array = np.array(sentiment_scores)
        neg_threshold1 = np.percentile(percentile_array, 25)
        pos_threshold1 = np.percentile(percentile_array, 75)
        neg_threshold2 = np.percentile(percentile_array, 33)
        pos_threshold2 = np.percentile(percentile_array, 67)
        n, bins, patches = plt.hist(x=sentiment_scores, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Sentiment')
        plt.ylabel('Frequency')
        plt.title('Sentiment Distribution')
        plt.axvline(neg_threshold2, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(pos_threshold2, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(neg_threshold1, color='g', linestyle='dashed', linewidth=1)
        plt.axvline(pos_threshold1, color='g', linestyle='dashed', linewidth=1)
        if self.retweets:
            plt.text(.4, 3000,"25rd percentile: "+str(neg_threshold1)+" \n33rd percentile: "+str(neg_threshold2)+" \n67th percentile: "+str(pos_threshold2)+" \n75th percentile: "+str(pos_threshold1))
        else:
            plt.text(.4, 1500,"25rd percentile: "+str(neg_threshold1)+" \n33rd percentile: "+str(neg_threshold2)+" \n67th percentile: "+str(pos_threshold2)+" \n75th percentile: "+str(pos_threshold1))
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()

    def geolocating(self):
        """Uses the geonames API to convert the user location to latitude and longitude coordinates"""
        j = 0 
        limit = app_config["api_rate_limits"]["geonames_api_rate_limit"]
        if j < limit:
            for i in range(len(self.tweet_df)):
                if self.tweet_df["user_location"][i] != "":
                    j=j+1
                    location = self.tweet_df["user_location"][i]
                    new_coords = location_2_coordinates(location, self.geolocator)
                    self.tweet_df.loc[i,["latitude", "longitude"]] = new_coords["latitude"], new_coords["longitude"]
                    # self.tweet_df["longitude"][i] = new_coords["longitude"]
        else:
            print("The Geonames API's maximum was reached. Only first "+ str(limit) +" locations were geolocated.")

    def coordinates_to_zip_county(self):
        """takes the latitude and logitude coordinates and searches information from the US Postal Service"""
        for i in range(len(self.tweet_df)):

            if self.tweet_df["latitude"][i] is None:
                self.tweet_df["latitude"][i] = np.nan
            if self.tweet_df["longitude"][i] is None:
                self.tweet_df["longitude"][i] = np.nan
            if self.tweet_df["zipcode"][i] is None:
                self.tweet_df["zipcode"][i] = np.nan
            # print("Longitude")
            # print(self.tweet_df["longitude"][i])
            # print("Zipcode")
            # print(self.tweet_df["zipcode"][i])
            if (math.isnan(self.tweet_df["latitude"][i])==False or self.tweet_df["latitude"][i] is not None) and (math.isnan(self.tweet_df["longitude"][i])==False or self.tweet_df["longitude"][i] is not None) and (math.isnan(self.tweet_df["zipcode"][i]) or self.tweet_df["zipcode"][i] is not None):
                print(i)
                output = coordinates_2_zip(self.search_zip, self.zip_fip_lookup, self.tweet_df["latitude"][i], self.tweet_df["longitude"][i], radius=2, returns=2)
                if output is None:
                    output = {"zipcode": None, "county": None, "city": None, "state": None, "city_state": None}
                self.tweet_df.loc[i,["zipcode", "county", "city", "state", "city_state"]] = output["zipcode"], output["county"], output["city"], output["state"], output["city_state"]
                # self.tweet_df["county"][i] = output["county"]
                # self.tweet_df["city"][i] = output["city"]
                # self.tweet_df["state"][i] = output["state"]
                # self.tweet_df["city_state"][i] = output["city_state"]

    def location_from_profile(self):
        for i in range(len(self.tweet_df)):
            print("validating tweets")
            if validate_tweet_location(self.tweet_df["user_location"][i]) and math.isnan(self.tweet_df["latitude"][i]) and math.isnan(self.tweet_df["longitude"][i]):
                print(i)
                match = validate_tweet_location(self.tweet_df["user_location"][i])
                temp = get_city_state(match, self.state_lookup)
                if temp is None:
                    pass
                else:
                    city = temp["city"]
                    state = temp["state"]
                    city_state = city+" "+state
                    results = location_2_everything(self.search_zip, city, state, self.zip_fip_lookup)
                    print("adding back to df")
                    self.tweet_df.loc[i, ["zipcode", "county", "city", "state", "city_state", "latitude", "longitude"]] = results["zipcode"], results["county"], city, state, city_state, results["latitude"], results["longitude"]
                    # self.tweet_df["county"][i] = results["county"]
                    # self.tweet_df["city"][i] = city
                    # self.tweet_df["state"][i] = state
                    # self.tweet_df["city_state"][i] = city_state
                    # self.tweet_df["latitude"][i] = results["latitude"]
                    # self.tweet_df["longitude"][i] = 

    def twitter_data_2_csv(self):
        """uses the pandas dataframe to write a csv"""
        csv_path = app_config["app"]["csv_path"]
        self.tweet_df.to_csv(csv_path, index=False, encoding='utf-8')

    def twitter_data_2_json(self):
        # NOTE: THIS IS A DUMB FUNCTION, BECAUSE THE DATA IS IN A JSON BEFORE IT GOES ITO A DATAFRAME SO TO JUST
        # CONVERT IT BACK AGAIN IS DUMB... JUST KEEP IT AS A JSON IN THE FIRST PLACE!!!!!!!
        """uses the pandas dataframe to write a csv"""
        json_path = app_config["app"]["json_path"]
        self.tweet_df.to_json(json_path)

    def tweet_df_2_pickle(self):
        """converts the pandas dataframe to a pickle file"""
        dataframe_pickle_path = app_config["app"]["dataframe_pickle_path"]
        self.tweet_df.to_pickle(dataframe_pickle_path)

    def run(self):
        """ This Method is how we would like the code to run when it is in production """
        if self.tweet_df.empty:
            self.tweets_to_df()
        self.coordinates_to_zip_county()
        self.location_from_profile()
        self.twitter_data_2_csv()
        self.twitter_data_2_json()

    def run_json_only(self):
        """ This Method is when I don't want the code to take 3 damn hours to run :) """
        if self.tweet_df.empty:
            self.tweets_to_df()
        self.twitter_data_2_json()

# if __name__ == '__main__':
#     df = pd.read_csv("dataframes/tweet_locations.csv")
#     ingest = TweetIngestionEngine(tweet_df=df)
#     print("Mapping Coordinates")
#     ingest.coordinates_to_zip_county()
#     print("Done!")
#     ingest.twitter_data_2_csv()
#     print("Extracting User Location")
#     ingest.location_from_profile()
#     print("Done!")
#     ingest.twitter_data_2_csv()

if __name__ == '__main__':
    print("creating tweet ingestion object")
    ingestor = TweetIngestionEngine()
    print("running tweet ingestion")
    ingestor.run_json_only()
