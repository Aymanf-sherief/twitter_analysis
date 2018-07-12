import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import nltk
import string
import textrank
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import datetime
from operator import itemgetter
import pickle
import time
import pymongo
from gensim.summarization.summarizer import summarize
from gensim.summarization.keywords import keywords
from sklearn.preprocessing import minmax_scale
from nltk.corpus import wordnet
from nltk.stem.porter import *

twitter = pymongo.MongoClient().twitter

#def current_milli_time(): return int(round(time.time() * 1000))


nltk.download('wordnet',download_dir = 'D:\\Programs\\Anaconda3\\share\\nltk_data')
nltk.download('stopwords',download_dir = 'D:\\Programs\\Anaconda3\\share\\nltk_data')

# with open('logit_4_14_vectorizer.pickle', 'rb') as f:
#     count_vectorizer = pickle.load(f)


# def predict_to_csv(topic, data):
#     data['sentiment'] = model.predict(
#         count_vectorizer.transform(data.clean_text))
#     data.to_csv(topic + "_" + str(current_milli_time()) + ".csv")
# def count_occurences(text, counts):
#    for word in text.split():
#        if len(word) > 3:

#            if word in counts:
#                counts[word]['weight'] += 1
#            else:
#                counts[word] = {'text': word, 'weight': 1}

#get_keywords = lambda text: keywords(text, scores = True, ratio = 1 if len(text) > 280 else len(text)/280.0)
#lemm = WordNetLemmatizer()
#tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

#stop = set(stopwords.words('english'))


# def clean_text(text):
#    regex = r'(@[^ ]*)' # remove user handles
#    regex += r'|(https?[^ ]*)' # remove links
#    regex += r'|(rt)'
#    cleaned_text = re.sub(regex, '', text.lower())
#    cleaned_text = ''.join([c for c in cleaned_text if c in
#    list(string.ascii_letters) + [' ']])
#    cleaned_text = ' '.join(lemm.lemmatize(word)
#                            for word in tknzr.tokenize(cleaned_text))
#    return cleaned_text.strip()

def get_keywords(text):
    words = 3
    raw_keywords = []
    while True:
        try:
            if words <= 0:
                return []
            raw_keywords = keywords(text, scores=True, words=words)
        except:
            words -= 1
            continue
        return raw_keywords


def get_text_counts(frame, topic):
    limit = int(.9*len(frame))
    text_counts = pd.Series(dtype=float)
    for index, tweet in frame.iloc[:limit].iterrows():
        try:

            phrases = get_keywords(tweet.clean_text)
            #print("phrases: ", phrases)
            #phrases = textrank.extractKeyphrases(tweet['clean_text'])
            for phrase, score in phrases:
                if phrase in text_counts:
                    text_counts[phrase] += score * \
                        float(np.log(tweet.popularity+1))
                else:
                    text_counts[phrase] = score * \
                        float(np.log(tweet.popularity+1))

        except (ZeroDivisionError, IndexError) as e:
            pass
        except Exception as e:
            print("ERROR: {}".format(e))

    to_del = set([topic]).union(set(stopwords.words('english'))
                                ).union(set(list(string.ascii_lowercase)))
    for phrase in text_counts.index.values:
        words = phrase.split()
        if len(words) > 1:
            for word in words:
                if word in text_counts or len(word.strip()) < 3:
                    to_del.add(word)

    for word in to_del:
        if word in text_counts.index.values:
            text_counts = text_counts.drop(word)

    #wordcloud_data = []
    # for text, count in text_counts.items():
    #    wordcloud_data.append({'text': text, 'count': count})

    return text_counts


def filter_words(pos_text_counts, neg_text_counts):
    for word in pos_text_counts.index.values:
        if word in neg_text_counts.index.values:
            if neg_text_counts[word] > pos_text_counts[word]:
                neg_text_counts[word] -= pos_text_counts[word]
                pos_text_counts[word] = 0
            else:
                pos_text_counts[word] -= neg_text_counts[word]
                neg_text_counts[word] = 0

    pos_text_counts = (pos_text_counts - pos_text_counts.min()) / \
        (pos_text_counts.max() - pos_text_counts.min())
    neg_text_counts = (neg_text_counts - neg_text_counts.min()) / \
        (neg_text_counts.max() - neg_text_counts.min())

    pos_wordcloud_data = []
    neg_wordcloud_data = []
    for text, count in pos_text_counts.items():
        pos_wordcloud_data.append({'text': text, 'count': count})

    for text, count in neg_text_counts.items():
        neg_wordcloud_data.append({'text': text, 'count': count})

    return pos_wordcloud_data, neg_wordcloud_data


def format_time(created_at):
    group = created_at.strftime("%Y-%m-%d %H:00:00")
    return str(group)


def get_timeseries_data(frame):

    time_series = {}
    frame['date'] = pd.to_datetime(frame.created_at).apply(format_time)
    group_by_date = frame.groupby('date')
    count_by_date = group_by_date['favorite_count'].aggregate('count')
    popularity_sum = group_by_date['popularity'].aggregate('sum')
    pop_timeseries = {'series': popularity_sum,
                      'type': 'scatter',
                      'visible': "legendonly"}

    time_series = {'series': count_by_date,
                   'type': 'scatter'
                   }

    return [time_series, pop_timeseries]


def add_vecs(a, b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c


def calculate_age(born):
    #born = datetime.datetime.fromtimestamp(born)
    today = datetime.date.today()
    age = today.year - born.year - \
        ((today.month, today.day) < (born.month, born.day))

    return age


def interval_to_label(intrvl):
    if intrvl.right >= 10:
        return('older than {}'.format(intrvl.left))
    return "{} - {}".format(intrvl.left, intrvl.right)


def get_account_ages_data(frame):

    datetime_bins = np.array([0, 1, 2, 3, 4, 6, 8, 10])

    frame['account_age'] = pd.to_datetime(
        frame['user_created_at']).apply(calculate_age)
    age_group = pd.cut(frame['account_age'],
                       bins=datetime_bins).apply(interval_to_label)
    age_counts = age_group.value_counts()
    print("type: ", type(age_counts))
    age_counts = age_counts.sort_index()
    account_ages = {
        'y': age_counts.values.tolist(),
        'x': age_counts.index.values.astype(str).tolist(),
        'type': 'bar',
        'marker': {'color': ''}
        # 'hole': .5
    }

    return account_ages


def get_topic_summary(frame):
    frame = frame.sort_values('popularity', ascending=False)
    limit = len(frame) if len(frame) < 100 else 100
    text = ". ".join(frame.clean_text.values[:limit])
    print("     summarizing {} tweets...".format(limit))
    summary = summarize(text)
    return summary


def data_from_result(topic, result):

    data = pd.DataFrame(result).drop_duplicates(subset='clean_text')
    print("found {} tweet".format(len(data)))

    print("getting timeseries data")
    time_series = get_timeseries_data(data)

    print("getting account ages data")
    account_ages = get_account_ages_data(data)

    print("getting wordcloud data")
    text_counts = get_text_counts(data, topic)

    print("getting topic summary")
    summary = get_topic_summary(data)

    data = {
        'time_series': time_series,
        'text_counts': text_counts,
        'account_ages': [account_ages],
        'summary': summary
    }
    return data


def get_dataframe(topic):
    topic = clean_text(topic)
    positive_results = list(twitter.tweets.find({'text': {"$regex": topic},
                                                 'sentiment': {'$eq': 1}}))
    print("positive len:", len(positive_results))
    negative_results = list(twitter.tweets.find({'text': {"$regex": topic},
                                                 'sentiment': {'$eq': 0}}))
    print("negative len:", len(negative_results))

    #frame = frame.sort_values('popularity', ascending = False)
    pos_dataframe = pd.DataFrame(positive_results).drop_duplicates(
        subset='clean_text').sort_values('popularity', ascending=False)
    neg_dataframe = pd.DataFrame(negative_results).drop_duplicates(
        subset='clean_text').sort_values('popularity', ascending=False)

    return pos_dataframe.set_index("_id"), neg_dataframe.set_index("_id")


def get_response(topic, positive_results, negative_results):

    print("getting positive data analysis")
    pos_data = data_from_result(topic, positive_results)
    print("-------------------------------------------", end="\n\n")

    print("getting negative data analysis")
    neg_data = data_from_result(topic, negative_results)
    print("-------------------------------------------", end="\n\n")

    print("filtering wordclouds")
    pos_data['wordcloud_data'], neg_data['wordcloud_data'] = filter_words(
        pos_data['text_counts'], neg_data['text_counts'])
    del pos_data['text_counts']
    del neg_data['text_counts']

    print("finalizing...")
    return pos_data, neg_data


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
