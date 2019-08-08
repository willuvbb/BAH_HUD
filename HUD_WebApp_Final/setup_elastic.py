from elasticsearch import helpers, Elasticsearch
import csv

es = Elasticsearch()

# delete the index "twitter", ignoring if it's already deleted/doesn't exist (ignore 404 and 400)
es.indices.delete(index='twitter', ignore=[400, 404])
# create the index
es.indices.create(index='twitter')

# Save the data to the index
with open('files/AllTweets_WithStates.csv') as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='twitter', doc_type='tweets')



# delete the index "states", ignoring if it's already deleted/doesn't exist (ignore 404 and 400)
es.indices.delete(index='states', ignore=[400, 404])

# create the index
es.indices.create(index='states')

# Save the data to the index
with open('files/StatesWithCounts.csv') as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='states', doc_type='states')