import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import re

def same_feature_sub(data):
    columns = data.columns
    home_columns = [col for col in columns if col.startswith('home_')]
    away_columns = [col for col in columns if col.startswith('away_')]


    common_suffixes = set([re.sub('home_', '', col) for col in home_columns]).intersection(
        set([re.sub('away_', '', col) for col in away_columns]))


    for suffix in common_suffixes:
        home_col = f'home_{suffix}'
        away_col = f'away_{suffix}'
        new_col = f'diff_{suffix}'
        data[new_col] = data[home_col] - data[away_col]
        print ("new_col: ", new_col)

    return data

def preprocess_data(data, is_train=True, label_encoder=None):
    if is_train:
        data = data.drop(['id', 'date', 'home_pitcher', 'away_pitcher', 'home_team_season', 
                          'away_team_season'], axis=1)
    else:
        data = data.drop(['id','home_pitcher', 'away_pitcher', 'home_team_season', 
                          'away_team_season'], axis=1)

    if is_train:
        X = data.drop('home_team_win', axis=1)
        y = data['home_team_win'].map({True: 1, False: 0})
    else:
        X = data
        y = None

    #trans to numeric
    X['is_night_game'] = X['is_night_game'].map({True: 1, False: 0})

    if is_train:
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(X['home_team_abbr'])

    X['abbr_home_team'] = label_encoder.transform(X['home_team_abbr'])
    X['abbr_away_team'] = label_encoder.transform(X['away_team_abbr'])
    X = X.drop(['home_team_abbr', 'away_team_abbr'], axis=1)

    #cleaning

    #feature engineering
    # X = same_feature_sub(X) # hint because seem not increse the accuracy

    return X, y, label_encoder


def train_model_catboost(X_train, y_train, X_val, y_val):
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(iterations=1000, eval_metric='Accuracy', verbose=100)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=True)
    plt.show()
    
    return model

def predict_and_save(model, X_test, test_data, filename='./data/results.csv'):
    y_pred = model.predict(X_test)
    res = pd.DataFrame()
    res['id'] = test_data['id']
    res['home_team_win'] = y_pred
    res['home_team_win'] = res['home_team_win'].map({1: True, 0: False})
    res.to_csv(filename, index=False)
