import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb


# Create features from   timestamps
click_data = pd.read_csv('input/feature-engineering-data/train_sample.csv',
                         parse_dates=['click_time'])
click_times = click_data['click_time']
clicks = click_data.assign(day=click_times.dt.day.astype('uint8'),
                           hour=click_times.dt.hour.astype('uint8'),
                           minute=click_times.dt.minute.astype('uint8'),
                           second=click_times.dt.second.astype('uint8'))

from itertools import combinations
from sklearn.preprocessing import LabelEncoder


label_enc = LabelEncoder()
cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)

# Iterate through each pair of features, combine them into interaction features
feature_interactions = list(combinations(cat_features, 2))

for feature_inter_pair in feature_interactions:
    label_enc = LabelEncoder()
    interactions_data = clicks[feature_inter_pair[0]].map(str) + '_' + clicks[feature_inter_pair[1]].map(str)
    feature_pair = '_'.join(tups for tups in feature_inter_pair)
    interactions[feature_pair] = label_enc.fit_transform(interactions_data)


def count_past_events(series):
    clicked = pd.Series(clicks.index, index=clicks.click_time, name='click_count').sort_index()
    count_past_events = clicked.rolling('6H').count() -1
    count_past_events.index = clicked.values
    count_past_events = count_past_events.reindex(clicks.index)
    return count_past_events

def previous_attributions(series):
    """Returns a series with the number of times an app has been downloaded."""
    clicked = pd.Series([clicks.index,clicks.is_attributed], index=clicks.click_time, name='click_count').sort_index()
    count_past_events = clicked.rolling('6H').count() - 1
    count_past_events.index = clicked.values
    count_past_events = count_past_events.reindex(clicks.index)
    return count_past_events
past_events = pd.read_parquet('input/feature-engineering-data/past_6hr_events.pqt')
time_deltas = pd.read_parquet('input/feature-engineering-data/time_deltas.pqt')

print(time_deltas)