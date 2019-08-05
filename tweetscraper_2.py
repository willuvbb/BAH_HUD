#!/usr/bin/env python
# coding: utf-8

# # Twitter Scraper
# 
# ### Author
# Last Modified by Will Cunningham on 6/19/19
# 
# Original script obtained on Github from Ritvikmath. https://github.com/ritvikmath/ScrapingData/blob/master/Scraping%20Twitter%20Data.ipynb
# 
# ### Function:
# The script below scrapes tweets by a user defined hashtag phrase.
#  
# ### Dependencies:
#  
# 1. A Twitter developer account
#     if you don't have one, here is where you can request one: https://developer.twitter.com
# 2. A Twitter user account
#    
# ### Keys and token info:
#  
# **Consumer API keys**
# - G91QOiv5k65dURn4f38HYmi9i (API key)
# - xhboh64wsUyHq1OfwgVc6EWpWJKGc3Ob9XSn7gK63J6fxPXtgg (API secret key)
#  
# **Access token & access token secret**
# - 1139543220004773889-QTxZITy6Q06Z1R9ElqEafSh6mjMp8q (Access token)
# - dSCaONBH8x00N1ftMUAW9xgmjd466t9TMMs213nxQGoTZ (Access token secret)
#  
# **Sources:
#     https://www.youtube.com/watch?v=Ou_floKQqd8&t=4s
#     https://github.com/ritvikmath/ScrapingData/blob/master/Scraping%20Twitter%20Data.ipynb
#    
# **For more info on tweepy:** http://docs.tweepy.org/en/v3.5.0/

# In[2]:


import json
import csv
import tweepy
import re

from api_keys import consumer_key, consumer_secret, access_token, access_token_secret

"""
INPUTS:
    consumer_key, consumer_secret, access_token, access_token_secret: codes 
    telling twitter that we are authorized to access this data
    hashtag_phrase: the combination of hashtags to search for
OUTPUTS:
    none, simply save the tweet info to a spreadsheet
"""

# This function searches via handles instead of hashtags  
def general_search(consumer_key, consumer_secret, access_token, access_token_secret, query, filename):
    
    #create authentication for accessing Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    #initialize Tweepy API
    api = tweepy.API(auth)
    
    #get the name of the spreadsheet we will write to
#     fname = '_'.join(re.findall(r"@(\w+)", handle_phrase))
    fname = filename

    #open the spreadsheet we will write to
    with open("./CSVloc/"+fname+".csv", 'w') as file:

        w = csv.writer(file)

        #write header row to spreadsheet
        w.writerow(['timestamp', 'tweet_text', 'username', 'all_hashtags', 'followers_count'])

        #for each tweet matching our hashtags, write relevant info to the spreadsheet
#         for tweet in tweepy.Cursor(api.search, q=handle_phrase+' -filter:retweets', \

        for tweet in tweepy.Cursor(api.search, q=query+' -filter:retweets',                                    lang="en", tweet_mode='extended').items(1000):
            w.writerow([tweet.created_at, tweet.full_text.replace('\n',' ').encode('utf-8'), tweet.user.screen_name.encode('utf-8'), [e['text'] for e in tweet._json['entities']['hashtags']], tweet.user.followers_count])    

        print('The file '+ filename + '.csv has been created in your working directory.')

    with open("./JSONloc/"+fname+".json", 'w') as f:
        for tweet in tweepy.Cursor(api.search, q=query+' -filter:retweets',                                    lang="en", tweet_mode='extended').items(1000):
#             print(type(tweet))
            f.write(json.dumps(tweet._json)+"\n")
        print('The file '+ filename + '.json has been created in your working directory.')



while 1:
    filename = input('Enter the file name with alphanum. and underscore (w/o extension): ')
    query = input('Enter Search Query: (Options: #, @ (mention), from: (from acct), or text: )')
    if __name__ == '__main__':
        general_search(consumer_key, consumer_secret, access_token, access_token_secret, query, filename)



