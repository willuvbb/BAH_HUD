from flask import Flask, render_template, request, jsonify
from elasticsearch import Elasticsearch

app = Flask(__name__)
es = Elasticsearch(port=9200)

@app.route('/')
def home():
    # Query the elastic search database
    res = es.search(
        index="twitter",
        size=1000,
        body={
            "query": {
                "match_all": {}
            }
        }
    )

    tweets_to_show = res['hits']['hits']
    # print(tweets_to_show)
    tabledata = []
    for tweet in tweets_to_show:
        tweet_dict = {
            "timestamp": tweet['_source']['timestamp'],
            "tweet": tweet['_source']['tweet'],
            "content": tweet['_source']['content'],
            "emotion": tweet['_source']['emotion'],
            "username": tweet['_source']['username'],
            "followers": tweet['_source']['followers'],
            "location": tweet['_source']['location'],
            "state": tweet['_source']['state']
        }
        tabledata.append(tweet_dict)

    # Query the elastic search database
    stateRes = es.search(
        index="states",
        size=52,
        body={
            "query": {
                "match_all": {}
            }
        }
    )

    states_preprocessing = stateRes['hits']['hits']
    # print(tweets_to_show)
    states = []
    for state in states_preprocessing:
        state_dict = {
            "State": state['_source']['State'],
            "Abbreviation": state['_source']['Abbreviation'],
            "count": state['_source']['count']
        }
        states.append(state_dict)

    return render_template('Dashboard.html', tweets=tabledata, states=states)

if __name__ == '__main__':
    app.run(port=5000)


