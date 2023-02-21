from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import CoClustering


def load_data(df):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user_id', 'song_id', 'play_count']], reader)
    return data


def grid_search_user(df):
    data = load_data(df)
    param_grid = {'k': [10, 20, 30], 'min_k': [3, 6, 9],
                  'sim_options': {'name': ["cosine", 'pearson', "pearson_baseline"],
                                  'user_based': [True], "min_support": [2, 4]}
                  }

    gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    return gs


def grid_search_item(df):
    data = load_data(df)
    param_grid = {'k': [10, 20, 30], 'min_k': [3, 6, 9],
                  'sim_options': {'name': ["cosine", 'pearson', "pearson_baseline"],
                                  'user_based': [False], "min_support": [2, 4]}
                  }

    gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    return gs


def grid_search_svd(df):
    data = load_data(df)
    param_grid = {'n_epochs': [10, 20, 30], 'lr_all': [0.001, 0.005, 0.01],
                  'reg_all': [0.2, 0.4, 0.6]}

    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    return gs


def grid_search_clustering(df):
    data = load_data(df)
    param_grid = {'n_cltr_u': [5, 6, 7, 8], 'n_cltr_i': [5, 6, 7, 8], 'n_epochs': [10, 20, 30]}

    gs = GridSearchCV(CoClustering, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    return gs
