"""
Routes and views for the flask application.
"""

from datetime import datetime
from GP_flask.utils import *
from GP_flask import app 
from flask import Flask, render_template, jsonify
import pymongo
from collections import Counter
import pandas as pd
import json
import os
import pickle
from flask import send_from_directory

model = pickle.load(open('D:\logit_4_14\\logit_4_14.pickle', 'rb'))
vectorizer = pickle.load(
    open('D:\logit_4_14\\logit_4_14_vectorizer.pickle', 'rb'))
print(type(model))

twitter = pymongo.MongoClient().twitter
positive_results = pd.DataFrame()
negative_results = pd.DataFrame()
polarities = []

@app.route("/")
def home():
    return render_template('home_search.html')


@app.route("/search/<topic>")
def search(topic):
    print("topic:", topic)
    topic = topic.lower()
    twitter.topics.update_one({'_id': topic}, {'$set': {'_id': topic}}, upsert=True)
    positive_results, negative_results = get_dataframe(topic)

    print("positive len:", len(positive_results))
    print("negative len:", len(negative_results))
    polarities = list(twitter.tweets.aggregate([{'$match': {'text': {"$regex": topic}}},
                                                {"$group": {"_id": None,
                                                            "positive_polarity_sum": {"$sum": '$positive_polarity'},
                                                            "negative_polarity_sum": {"$sum": '$negative_polarity'}
                                                            }}]))[0]

    #pos_data, neg_data = get_response(topic, positive_results,
    #negative_results)
    #data = {'negative': neg_data, 'positive': pos_data, 'polarities':
    #polarities}
    return get_time_series(topic)

@app.route("/search/timeseries/<topic>")
def get_time_series(topic):
    positive_results, negative_results = get_dataframe(topic)
    neg_timeseries = get_timeseries_data(negative_results)
    pos_timeseries = get_timeseries_data(positive_results)

    reach_series = pos_timeseries[1]['series'].add(neg_timeseries[1]['series'], fill_value = 0)
    reach_ts = { 'x': reach_series.index.values.tolist(),
                  'y': reach_series.values.tolist(),
                  'type': 'scatter',
                  'name': "total reach",
                  'visible': "legendonly" }

    total_series = pos_timeseries[0]['series'].add(neg_timeseries[0]['series'], fill_value = 0)
    total_ts = { 'x': total_series.index.values.tolist(),
                  'y': total_series.values.tolist(),
                  'type': 'scatter',
                  'name': "total trend",
                  'visible': "legendonly" }
                  


    pos_timeseries[0]['name'] = 'positive trend (No. Tweets)'
    pos_timeseries[0]['line'] = { 'color': '#008000' }
    pos_timeseries[0]['x'] =  pos_timeseries[0]['series'].index.values.tolist()
    pos_timeseries[0]['y'] =  pos_timeseries[0]['series'].values.tolist()
    del pos_timeseries[0]['series']
    pos_timeseries[1]['name'] = 'positive reach (No. People)'
    pos_timeseries[1]['line'] = { 'color': '#006400' }
    pos_timeseries[1]['x'] =  pos_timeseries[1]['series'].index.values.tolist()
    pos_timeseries[1]['y'] =  pos_timeseries[1]['series'].values.tolist()
    del pos_timeseries[1]['series']
    neg_timeseries[0]['name'] = 'negative trend (No. Tweets)'
    neg_timeseries[0]['line'] = { 'color': '#E50000' }
    neg_timeseries[0]['x'] =  neg_timeseries[0]['series'].index.values.tolist()
    neg_timeseries[0]['y'] =  neg_timeseries[0]['series'].values.tolist()
    del neg_timeseries[0]['series']
    neg_timeseries[1]['name'] = 'negative reach (No. People)'
    neg_timeseries[1]['line'] = { 'color': '#8b0000' }
    neg_timeseries[1]['x'] =  neg_timeseries[1]['series'].index.values.tolist()
    neg_timeseries[1]['y'] =  neg_timeseries[1]['series'].values.tolist()
    del neg_timeseries[1]['series']
    

    return json.dumps(pos_timeseries + neg_timeseries + [total_ts, reach_ts])

@app.route("/search/group/<topic>")
def get_group(topic):
    positive_results, negative_results = get_dataframe(topic)
    neg_account_ages = get_account_ages_data(negative_results)
    pos_account_ages = get_account_ages_data(positive_results)
    
    neg_account_ages['marker']['color'] = 'rgb(0,128,0)'
    pos_account_ages['marker']['color'] = 'rgb(229,0,0)'

    pos_account_ages['name'] = 'Positive'
    neg_account_ages['name'] = 'Negative'

    pos_hist = {'x': positive_results.popularity.values.tolist(),
                'type': 'histogram',
                'opacity': .5,
                'name': 'Positive',
                'marker':{'color': 'rgb(0,128,0)'}}
    neg_hist = {'x': negative_results.popularity.values.tolist(),
                'type': 'histogram',
                'opacity': .5,
                'name': 'Negative',
                'marker':{'color': 'rgb(229,0,0)'}}

    pos_box = {"x": positive_results.popularity.values.tolist(),
                "type": "box",
                "name": "Positive",
                'marker':{'color': 'rgb(0,128,0)'}}

    neg_box = {"x": negative_results.popularity.values.tolist(),
                "type": "box",
                "name": "Negative",
                'marker':{'color': 'rgb(229,0,0)'}}

    return json.dumps({'account_ages': [neg_account_ages, pos_account_ages],
                       'popularity_hist': [pos_hist, neg_hist],
                       "box": [neg_box, pos_box]})


@app.route("/search/wordcloud/<topic>")
def get_wordcloud_data(topic):
    positive_results, negative_results = get_dataframe(topic)
    neg_textcounts = get_text_counts(negative_results, topic)
    pos_textcounts = get_text_counts(positive_results, topic)

    pos_wordcloud_data, neg_wordcloud_data = filter_words(pos_textcounts, neg_textcounts)

    return json.dumps({"negative_wordcloud": neg_wordcloud_data,
                       "positive_wordcloud": pos_wordcloud_data})


@app.route("/search/summary/<topic>")
def get_summary(topic):
    positive_results, negative_results = get_dataframe(topic)
    neg_summary = get_topic_summary(negative_results)
    pos_summary = get_topic_summary(positive_results)

    return json.dumps({"negative_summary": neg_summary,
                       "positive_summary": pos_summary})


@app.route("/search/toptweets/<topic>")
def get_top_tweets(topic):
    positive_results, negative_results = get_dataframe(topic)

    return json.dumps({'pos_tweets': positive_results.index.values.tolist(),
                       'neg_tweets': negative_results.index.values.tolist()})

@app.route('/fonts/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.getcwd())
    print(root_dir)
    full_dir = os.path.join(root_dir, 'GP_twitter_analysis','GP_flask', 'static', 'fonts')
    print(full_dir)
    return send_from_directory(full_dir, filename)


@app.route("/api/agg/<topic>/<raw_func>")
def agg(topic, raw_func):
    positive_results, negative_results = get_dataframe(topic)
    print(raw_func)

    try:
        func = json.loads(raw_func)
    except json.JSONDecodeError:
        func = raw_func
    except:
        return json.dumps({"error": "wrong function format"})

    pos_response = positive_results.agg(func).to_json()
    neg_response = negative_results.agg(func).to_json()

    return json.dumps({"negative results": neg_response,
            "positive results": pos_response})


@app.route("/api/select/<topic>/<since_YYYY_MM_DD>")
def select(topic, since_YYYY_MM_DD):
    positive_results, negative_results = get_dataframe(topic)

    if since_YYYY_MM_DD.strip() != '0':
        try:
            since = pd.to_datetime(since_YYYY_MM_DD)
        except:
            return json.dumps({"error": "wrong date format, use YYYY-MM/DD"})
            
        print(since)
        positive_results.created_at = pd.to_datetime(positive_results.created_at)
        negative_results.created_at = pd.to_datetime(negative_results.created_at)

        positive_results = positive_results[positive_results.created_at > since].sort_values('created_at', ascending = False)
        negative_results = negative_results[negative_results.created_at > since].sort_values('created_at', ascending = False)

    # add_column = lambda row, name: row 
    neg_response = negative_results["_full_tweet_"].tolist()
    pos_response = positive_results["_full_tweet_"].tolist()
    print(negative_results.index.dtype)
    for field in neg_response[0]:
        print("     field: {} --- type: {}".format(field, type(neg_response[0][field])))
    for indx, tweet in enumerate(neg_response):
        neg_response[indx]['sentiment'] = int(negative_results.loc[str(tweet["id"])].sentiment)
        neg_response[indx]['positive_polarity'] = float(negative_results.loc[str(tweet["id"])].negative_polarity)
        neg_response[indx]['positive_polarity'] = float(negative_results.loc[str(tweet["id"])].positive_polarity)

    for indx, tweet in enumerate(pos_response):
        pos_response[indx]['sentiment'] = int(positive_results.loc[str(tweet["id"])].sentiment)
        pos_response[indx]['positive_polarity'] = float(positive_results.loc[str(tweet["id"])].negative_polarity)
        pos_response[indx]['positive_polarity'] = float(positive_results.loc[str(tweet["id"])].positive_polarity)


    return json.dumps(neg_response + pos_response)

@app.route("/api/text/<text>")
def analyze_text(text):

    tweet_clean_text = clean_text(text)
    pred_text = clean_text_prediction(tweet_clean_text)
    pred_feats = vectorizer.transform([pred_text])
    polarities = model.predict_proba(pred_feats)[0]

    sentiment = int(polarities[1] > polarities[0])
    topics = list(twitter.topics.find())
    possible_topics = [topic['_id'] for topic in topics if topic['_id'] in tweet_clean_text.split()]
    keywords_raw = get_keywords(tweet_clean_text)
    keywords = []
    for phrase, score in keywords_raw:
        keywords.append({"phrase": phrase,
                         "score": score})



    return json.dumps({
        "sentiment": sentiment,
        "positive polarity": polarities[1],
        "negative polarity": polarities[0],
        "topics": possible_topics,
        "keywords": keywords
        })

