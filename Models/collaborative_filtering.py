import pandas as pd
import numpy as np
from .rank_based import tally_average_playcounts
from collections import defaultdict
from surprise import accuracy
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import CoClustering


def split_dataset(df, test_size, random_state):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user_id', 'song_id', 'play_count']], reader)
    train, test = train_test_split(data, test_size, random_state)
    return train, test


def precision_recall_at_k(model, k=30, threshold=1.5):
    trainset, testset = split_dataset(test_size=0.4, random_state=42)
    user_est_true = defaultdict(list)
    predictions = model.test(testset)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[: k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[: k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    precision = round((sum(prec for prec in precisions.values()) / len(precisions)), 3)
    recall = round((sum(rec for rec in recalls.values()) / len(recalls)), 3)

    accuracy.rmse(predictions)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F_1 score: ', round((2 * precision * recall) / (precision + recall), 3))


def user_user_model():
    trainset, testset = split_dataset(test_size=0.4, random_state=42)
    sim_options = {'name': 'cosine', 'user_based': True}
    model = KNNBasic(random_state=1, sim_options=sim_options, verbose=False)
    model.fit(trainset)
    precision_recall_at_k(model)
    return model


def item_item_model():
    trainset, testset = split_dataset(test_size=0.4, random_state=42)
    sim_options = {'name': 'cosine', 'user_based': False}
    model = KNNBasic(random_state=1, sim_options=sim_options, verbose=False)
    model.fit(trainset)
    precision_recall_at_k(model)
    return model


def matrix_factorization_model():
    trainset, testset = split_dataset(test_size=0.4, random_state=42)
    model = SVD(random_state=1)
    model.fit(trainset)
    precision_recall_at_k(model)
    return model


def clustering_model():
    trainset, testset = split_dataset(test_size=0.4, random_state=42)
    model = CoClustering(random_state=1)
    model.fit(trainset)
    precision_recall_at_k(model)
    return model


def get_recommendations(data, user_id, top_n, algo):
    recommendations = []
    user_item_interactions_matrix = data.pivot_table(index='user_id', columns='song_id', values='play_count')
    non_interacted_products = user_item_interactions_matrix.loc[user_id][
        user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

    for item_id in non_interacted_products:
        est = algo.predict(user_id, item_id).est
        recommendations.append((item_id, est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]


def ranking_songs(df, recommendations):
    final_playbacks = tally_average_playcounts(df)
    ranked_songs = \
        final_playbacks.loc[[items[0] for items in recommendations]].sort_values('play_freq', ascending=False)[
            ['play_freq']].reset_index()

    ranked_songs = ranked_songs.merge(pd.DataFrame(recommendations, columns=['song_id', 'predicted_playcounts']),
                                      on='song_id', how='inner')

    ranked_songs['corrected_playcounts'] = ranked_songs['predicted_playcounts'] - 1 / np.sqrt(ranked_songs['play_freq'])

    ranked_songs = ranked_songs.sort_values(by='corrected_playcounts', ascending=False)
    return ranked_songs
