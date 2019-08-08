import pandas as pd
# Importing the dataset
# dataset = pd.read_csv('LaudanTweets.csv', quoting = 3)
input_data_1 = pd.read_csv('LaudanTweets.csv')
input_data_2 = pd.read_csv('LaudanTweets_2.csv')

# # Take out the blank rows
# input_data_1 = input_data_1.dropna(axis=0,how='all')
# input_data_2 = input_data_2.dropna(axis=0,how='all')

# # Take only the columns that we want
# input_data_1 = input_data_1[['tweet_text','Primary','Secondary']]
# input_data_2 = input_data_2[['tweet_text','Primary','Secondary']]

dataset = pd.concat([input_data_1,input_data_2])

##### PREPROCESSING #####
# Take only the columns that we want
dataset = dataset[['tweet_text','Primary','Secondary']]
# Take out the blank rows
dataset = dataset.dropna(axis=0,how='all')
# Reset the indexes
dataset = dataset.reset_index(drop=True)


dataset.shape
input_data_1.shape
input_data_2.shape