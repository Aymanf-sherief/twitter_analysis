# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:54:03 2018

@author: noura
"""
import pandas as pd
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import pymongo
import json
import pprint
from threading import Timer
import pickle
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import string
import numpy as np
import _pickle as pickle

model = pickle.load(open('D:\logit_4_14\\logit_4_14.pickle', 'rb'))
vectorizer = pickle.load(
    open('D:\logit_4_14\\logit_4_14_vectorizer.pickle', 'rb'))
print(type(model))
# Variables that contains the user credentials to access Twitter API
access_token = "1464919340-4OHUBQdSJjTFG8LqMPmstkWPj0qPlemiQic4FmV"
access_token_secret = "8MdNlE3Q9N1NCoj9XfsiZaws43dJXoI0UMnoEGn723pkz"
consumer_key = "eKytO5h04YQhqwczDEXc0ubay"
consumer_secret = "WktfrnSWTSzBKwEoiooAPIQ11SAb09yjfsIO3KcClYRQsgb36b"
twitter = pymongo.MongoClient().twitter
# This is a basic listener that just prints received tweets to stdout.


ps = PorterStemmer()

nltk.download('wordnet')
lemm = WordNetLemmatizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)


def clean_text(text):
    regex = r'(@[^ ]*)'  # remove user handles
    regex += r'|(https?[^ ]*)'  # remove links
    regex += r'|(rt)'
    cleaned_text = re.sub(regex, '', text.lower(), re.MULTILINE)
    cleaned_text = ''.join(
        [c for c in cleaned_text if c in list(string.ascii_letters) + [' ', '#']])
    #cleaned_text = ' '.join( ps.stem(word) for word in tknzr.tokenize(cleaned_text))
    return cleaned_text.strip()


def clean_text_prediction(cleaned_text):
    cleaned_text = ''.join(
        [c for c in cleaned_text if c in list(string.ascii_letters) + [' ']])
    cleaned_text = ' '.join(ps.stem(word)
                            for word in tknzr.tokenize(cleaned_text))

    return cleaned_text


class StdOutListener(StreamListener):

    def on_data(self, data):
        try:
            data = json.loads(data)
            # pprint.pprint (data)
            tweet_clean_text = clean_text(data['text'])
            pred_text = clean_text_prediction(tweet_clean_text)
            pred_feats = vectorizer.transform([pred_text])
            polarities = model.predict_proba(pred_feats)[0]

            tweet = {'text': data['text'].lower(),
                     'clean_text': tweet_clean_text,
                     'sentiment': int(np.argmax(polarities)),
                     'negative_polarity': polarities[0],
                     'positive_polarity': polarities[1],
                     'location': data['user']['location'],
                     'time_zone': data['user']['time_zone'],
                     'popularity': data['user']['followers_count'],
                     'user_created_at': data['user']['created_at'],
                     'verified': data['user']['verified'],
                     "favorite_count": data['favorite_count'],
                     "retweet_count": data['retweet_count'],
                     "reply_count": data['reply_count'],
                     "quote_count": data['quote_count'],
                     "coordinates": data['coordinates'],
                     'created_at': data['created_at'],
                     '_id': data['id_str'],
                     "_full_tweet_": data}

            # pprint.pprint (data)
            twitter.tweets.insert_one(tweet)
            return True
        except:

            print('error occured, passing')

    def on_error(self, status):
        print(status)


def update_track_list():
    global track_list
    track_list = twitter.topics.distinct('_id')


if __name__ == '__main__':

    print('starting')
    # This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l, async=True)

    global track_list
    track_list = twitter.topics.distinct('_id')
    pprint.pprint(track_list)

    track_list_updater = Timer(5 * 60, update_track_list)
    track_list_updater.start()
    while True:
        try:
            stream.filter(track=track_list, languages=["en"])
        except:
            print("streaming error, continuing")
            continue
