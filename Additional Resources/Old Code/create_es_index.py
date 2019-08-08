from elasticsearch import helpers, Elasticsearch
import csv

es = Elasticsearch()

es.indices.create(index = 'twitter')