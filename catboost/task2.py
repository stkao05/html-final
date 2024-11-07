from mlb import preprocess_data, train_model_catboost, predict_and_save
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # load data
    train_data = pd.read_csv('./data/task2/train_data.csv')
    X, y, label_encoder = preprocess_data(train_data)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=64)
    
    test_data = pd.read_csv('./data/task2/2024_test_data.csv')
    X_test, _, _ = preprocess_data(test_data, is_train=False, label_encoder=label_encoder)
    

    # train model
    model = train_model_catboost(X_train, y_train, X_val, y_val)

    # inference
    predict_and_save(model, X_test, test_data, './data/task2_results.csv')