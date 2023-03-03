import pandas as pd
import pickle
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from collections import defaultdict
from surprise import accuracy
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import CoClustering


def tally_average_playcounts(df):
    avg_count = df.groupby(by="song_id").mean()['play_count']
    play_freq = df.groupby(by="song_id").count()['play_count']
    final_play = pd.DataFrame({'avg_count': avg_count, 'play_freq': play_freq})
    pickle.dump(final_play, open('../music-recommendations/Models/play_frequencies.pkl', 'wb'))


def build_title_dictionary(df):
    id_lookup = pd.Series(df['title'].values, index=df['song_id']).to_dict()
    pickle.dump(id_lookup, open('../music-recommendations/Models/title_dictionary.pkl', 'wb'))


def assemble_text_features(df):
    df['text'] = df['title'] + ' ' + df['release'] + ' ' + df['artist_name']
    df = df[['user_id', 'song_id', 'play_count', 'title', 'text']]
    df = df.drop_duplicates(subset=['title'])
    df = df.set_index('title')
    return df


def tokenize(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in stopwords.words('english')]
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]
    return text_lems


def build_song_tfidf(df):
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    song_tfidf = tfidf.fit_transform(df['text'].values).toarray()
    return song_tfidf


def compute_similarities(song_tfidf):
    return cosine_similarity(song_tfidf, song_tfidf)


def precision_recall_at_k(testset, model, k=30, threshold=1.5):
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


def user_user_model(train, test):
    sim_options = {'name': 'pearson_baseline', 'user_based': True, 'min_support': 2}
    model = KNNBasic(random_state=1, sim_options=sim_options, k=30, min_k=9, verbose=False)
    model.fit(train)
    precision_recall_at_k(test, model)
    return model


def item_item_model(train, test):
    sim_options = {'name': 'pearson_baseline', 'user_based': False, 'min_support': 2}
    model = KNNBasic(random_state=1, sim_options=sim_options, k=20, min_k=6, verbose=False)
    model.fit(train)
    precision_recall_at_k(test, model)
    return model


def matrix_factorization_model(train, test):
    model = SVD(n_epochs=30, lr_all=0.01, reg_all=0.2, random_state=1)
    model.fit(train)
    precision_recall_at_k(test, model)
    return model


def clustering_model(train, test):
    model = CoClustering(n_cltr_u=5, n_cltr_i=5, n_epochs=10, random_state=1)
    model.fit(train)
    precision_recall_at_k(test, model)
    return model


def train_rank_based(df):
    tally_average_playcounts(df)
    build_title_dictionary(df)


def train_content_based(df):
    tfidf_df = df.copy()
    tfidf_df = assemble_text_features(tfidf_df)
    pickle.dump(tfidf_df, open('../music-recommendations/Models/songs_with_text_feature.pkl', 'wb'))

    tfidf = build_song_tfidf(tfidf_df)
    similar_songs = compute_similarities(tfidf)
    pickle.dump(similar_songs, open('../music-recommendations/Models/tfidf_similarities.pkl', 'wb'))


def train_collaborative_filtering(df):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user_id', 'song_id', 'play_count']], reader)
    trainset, testset = train_test_split(data, test_size=0.4, random_state=42)
    print('user-user model: ')
    user_model = user_user_model(trainset, testset)
    pickle.dump(user_model, open('../music-recommendations/Models/user_user.pkl', 'wb'))
    print('item-item model: ')
    item_model = item_item_model(trainset, testset)
    pickle.dump(item_model, open('../music-recommendations/Models/item_item.pkl', 'wb'))
    print('SVD model: ')
    svd_model = matrix_factorization_model(trainset, testset)
    pickle.dump(svd_model, open('../music-recommendations/Models/svd.pkl', 'wb'))
    print('clustering-based model: ')
    clustering = clustering_model(trainset, testset)
    pickle.dump(clustering, open('../music-recommendations/Models/clustering_based.pkl', 'wb'))


def main():
    df = pd.read_csv("../music-recommendations/Data/playbacks.csv")
    pickle.dump(df, open('../music-recommendations/Models/playbacks.pkl', 'wb'))
    train_rank_based(df)
    train_content_based(df)
    train_collaborative_filtering(df)


if __name__ == "__main__":
    main()
