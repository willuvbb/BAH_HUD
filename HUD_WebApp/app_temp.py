from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from elasticsearch import Elasticsearch
import json

app = Flask(__name__)
es = Elasticsearch(port=9200)

tweets = pd.read_csv('files/AllTweets_WithStates.csv')

states = pd.read_csv('files/StatesWithCounts.csv')


# tweets_as_json_list = []
# for index, row in tweets.iterrows():
#     temp_json = row.to_json()
#     tweets_as_json_list.append(temp_json)

tweets_json = tweets.to_json(orient='records',lines=True)
print(tweets_json)
print(type(tweets_json))

@app.route('/')
def home_page():
    return render_template('Dashboard.html', tabledata=tweets_json)


if __name__ == '__main__':
    app.run(port=5001)


