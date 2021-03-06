from elasticsearch import helpers, Elasticsearch
import csv

es = Elasticsearch()

with open('AllTweets.csv') as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='twitter', doc_type='tweets')