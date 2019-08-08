import pickle
with open('TweetDataset.pkl', 'rb') as f:
    tweets = pickle.load(f)

prim_sort = tweets.sort_values(by=['Primary'])

seco_sort = tweets.sort_values(by=['Secondary'])

seco_prim_sort = tweets.sort_values(by=['Secondary','Primary'])

prim_seco_sort = tweets.sort_values(by=['Primary','Secondary'])